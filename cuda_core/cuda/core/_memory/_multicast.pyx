# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.string cimport memcpy, memset
from libcpp.vector cimport vector

cimport cpython

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_check_open, Buffer_from_deviceptr_handle
from cuda.core._rt cimport (
    ContextHandle,
    DevicePtrHandle,
    MemAllocationHandle,
    MulticastBindingHandle,
    StreamHandle,
    VaMappingHandle,
    VaReservationHandle,
    VmmRangeHandle,
    as_cu,
    create_context_bound_legacy_stream,
    create_multicast_binding_handle,
    create_multicast_handle,
    create_va_mapping_handle_with_access,
    create_va_reservation_handle,
    create_vmm_range,
    deallocation_stream,
    deviceptr_create_vmm,
    get_last_error,
    get_primary_context,
    import_mem_allocation_handle,
    mem_allocation_is_imported,
    mem_allocation_size,
    set_deallocation_stream,
    va_mapping_allocation,
    vmm_range,
    vmm_range_mappings,
)
from cuda.core._stream cimport Stream, Stream_accept, Stream_handle_is_default_token, Stream_is_default_token
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._utils.version cimport cy_driver_version

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cuda.core._device import Device
from cuda.core._memory._virtual_memory_resource import VirtualMemoryBuffer
from cuda.core.typing import (
    VirtualMemoryGranularityType,
    VirtualMemoryHandleType,
    VirtualMemoryLocationType,
)

if TYPE_CHECKING:
    from cuda.core.graph import GraphBuilder

__all__ = [
    "MulticastBinding",
    "MulticastBuffer",
    "MulticastObject",
    "MulticastObjectDescriptor",
    "MulticastObjectOptions",
]

_FABRIC_HANDLE_SIZE = 64
_HANDLE_TYPES = frozenset({None, VirtualMemoryHandleType.FABRIC})


cdef int _require_multicast_driver() except -1:
    if cy_driver_version() < (12, 1, 0):
        raise NotImplementedError(
            "multicast objects need a CUDA 12.1 or newer driver "
            f"(current driver: {'.'.join(map(str, cy_driver_version()))})"
        )
    return 0


cdef int _raise_last_error() except -1:
    HANDLE_RETURN(get_last_error())
    raise RuntimeError(
        "internal cuda.core error, please report: a multicast handle factory "
        "returned an empty handle without recording a CUDA error"
    )


cdef int _raise_last_error_noting(str detail) except -1:
    try:
        _raise_last_error()
    except Exception as exc:
        exc.add_note(detail)
        raise
    return 0


cdef bint _stream_is_capturing(cydriver.CUstream s) except -1:
    cdef cydriver.CUstreamCaptureStatus cap_status
    IF CUDA_CORE_BUILD_MAJOR >= 13:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(s, &cap_status, NULL, NULL, NULL, NULL, NULL))
    ELSE:
        HANDLE_RETURN(cydriver.cuStreamGetCaptureInfo(s, &cap_status, NULL, NULL, NULL, NULL))
    return cap_status == cydriver.CU_STREAM_CAPTURE_STATUS_ACTIVE


def _normalize_handle_type(value) -> VirtualMemoryHandleType | None:
    if value is None:
        return None
    if not isinstance(value, VirtualMemoryHandleType):
        try:
            value = VirtualMemoryHandleType(value)
        except (ValueError, TypeError):
            raise ValueError(f"invalid handle_type {value!r}") from None
    if value not in _HANDLE_TYPES:
        raise ValueError(
            f"handle_type {str(value)!r} is not supported for multicast objects; use 'fabric' or None"
        )
    return value


def _normalize_granularity(value) -> VirtualMemoryGranularityType:
    if not isinstance(value, VirtualMemoryGranularityType):
        try:
            value = VirtualMemoryGranularityType(value)
        except (ValueError, TypeError):
            raise ValueError(f"invalid granularity {value!r}") from None
    return value


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


cdef cydriver.CUmemAllocationHandleType _driver_handle_type(object handle_type):
    if handle_type is None:
        return cydriver.CU_MEM_HANDLE_TYPE_NONE
    return cydriver.CU_MEM_HANDLE_TYPE_FABRIC


cdef void _fill_prop(
    cydriver.CUmulticastObjectProp* prop, int num_devices, size_t size, object handle_type
) noexcept:
    memset(prop, 0, sizeof(cydriver.CUmulticastObjectProp))
    prop.numDevices = num_devices
    prop.size = size
    prop.handleTypes = <unsigned long long>_driver_handle_type(handle_type)
    prop.flags = 0


cdef size_t _query_granularity(int num_devices, object handle_type, object granularity) except? 0:
    cdef cydriver.CUmulticastObjectProp prop
    cdef size_t gran = 0
    cdef cydriver.CUmulticastGranularity_flags flag = (
        cydriver.CU_MULTICAST_GRANULARITY_RECOMMENDED
        if granularity == VirtualMemoryGranularityType.RECOMMENDED
        else cydriver.CU_MULTICAST_GRANULARITY_MINIMUM
    )
    _require_multicast_driver()
    _fill_prop(&prop, num_devices, 0, handle_type)
    with nogil:
        HANDLE_RETURN(cydriver.cuMulticastGetGranularity(&gran, &prop, flag))
    return gran


cdef int _device_ordinal(object device) except -1:
    return Device(device).device_id


@dataclass
class MulticastObjectOptions:
    """Configuration of a :class:`MulticastObject`.

    Attributes
    ----------
    num_devices : int
        Number of devices that will join the multicast object with
        :meth:`MulticastObject.add_device`. Binding memory blocks until all of
        them have joined.
    size : int
        Requested size in bytes. It is rounded up to the granularity.
    handle_type : :obj:`~_memory.VirtualMemoryHandleType` | str | None
        ``"fabric"`` (the default) to share the multicast object with other
        processes through :attr:`MulticastObject.ipc_descriptor`, or `None`
        for a multicast object used by one process. Fabric handles need an
        NVLink fabric and an accessible IMEX channel.
    granularity : :obj:`~_memory.VirtualMemoryGranularityType` | str
        Which multicast granularity ``size`` is rounded to. The address
        reservation of :meth:`MulticastObject.map` uses the same value.
    """

    num_devices: int
    size: int
    handle_type: VirtualMemoryHandleType | str | None = VirtualMemoryHandleType.FABRIC
    granularity: VirtualMemoryGranularityType | str = VirtualMemoryGranularityType.RECOMMENDED

    def __post_init__(self) -> None:
        _positive_int("num_devices", self.num_devices)
        _positive_int("size", self.size)
        self.handle_type = _normalize_handle_type(self.handle_type)
        self.granularity = _normalize_granularity(self.granularity)


class MulticastObjectDescriptor:
    """Plain-data descriptor of a :class:`MulticastObject` exported with a fabric handle.

    Carries the opaque 64-byte fabric handle, the multicast object's size and
    the number of devices it was created for. Plain ``pickle``,
    ``multiprocessing``, or a transport of your own can carry it; rebuild one
    received outside cuda.core with the constructor, then import it with
    :meth:`MulticastObject.from_ipc_descriptor`.

    Nothing is pinned: the exporting process must keep its
    :class:`MulticastObject` (or a binding or mapping made from it) alive
    until every importer has imported it.

    Parameters
    ----------
    handle : bytes
        The opaque 64-byte fabric handle.
    size : int
        Size of the multicast object in bytes.
    num_devices : int
        Number of devices the multicast object was created for.
    """

    __slots__ = ("_handle", "_size", "_num_devices")

    def __init__(self, handle: bytes, size: int, num_devices: int) -> None:
        handle = bytes(handle)
        if len(handle) != _FABRIC_HANDLE_SIZE:
            raise ValueError(f"a fabric handle is {_FABRIC_HANDLE_SIZE} bytes, got {len(handle)}")
        self._handle = handle
        self._size = _positive_int("size", size)
        self._num_devices = _positive_int("num_devices", num_devices)

    @property
    def handle(self) -> bytes:
        """The opaque 64-byte fabric handle."""
        return self._handle

    @property
    def size(self) -> int:
        """Size of the multicast object in bytes."""
        return self._size

    @property
    def num_devices(self) -> int:
        """Number of devices the multicast object was created for."""
        return self._num_devices

    @property
    def handle_type(self) -> VirtualMemoryHandleType:
        """Always ``"fabric"``."""
        return VirtualMemoryHandleType.FABRIC

    def __reduce__(self) -> tuple[object, ...]:
        return type(self), (self._handle, self._size, self._num_devices)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MulticastObjectDescriptor):
            return NotImplemented
        return (self._handle, self._size, self._num_devices) == (
            other._handle, other._size, other._num_devices)

    def __hash__(self) -> int:
        return hash((self._handle, self._size, self._num_devices))

    def __repr__(self) -> str:
        return f"<MulticastObjectDescriptor size={self._size} num_devices={self._num_devices}>"


cdef class MulticastBinding:
    """Physical memory bound into a :class:`MulticastObject`.

    Returned by :meth:`MulticastObject.bind`. The binding keeps the multicast
    object and the bound memory alive; closing it, or releasing its last
    reference, unbinds the memory with ``cuMulticastUnbind``. The unbind is
    not stream-ordered: finish the work that reaches the memory through the
    multicast address first, for example by closing the
    :class:`MulticastBuffer` from :meth:`MulticastObject.map`, which waits
    for its stream.
    """

    cdef:
        vector[MulticastBindingHandle] _bindings
        size_t _offset
        size_t _size
        int _device_id

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("MulticastBinding objects are created by MulticastObject.bind()")

    @property
    def offset(self) -> int:
        """Offset of the binding in the multicast object, in bytes."""
        return self._offset

    @property
    def size(self) -> int:
        """Number of bound bytes."""
        return self._size

    @property
    def device_id(self) -> int:
        """Ordinal of the device that owns the bound memory."""
        return self._device_id

    @property
    def is_closed(self) -> bool:
        """Whether the binding was closed."""
        return self._bindings.empty()

    def close(self) -> None:
        """Unbind the memory. Calling it again does nothing."""
        self._bindings.clear()

    def __enter__(self) -> MulticastBinding:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __reduce__(self):
        raise TypeError("a MulticastBinding cannot be pickled")

    def __repr__(self) -> str:
        state = "closed" if self._bindings.empty() else f"offset={self._offset} size={self._size}"
        return f"<MulticastBinding device={self._device_id} {state}>"


cdef class MulticastBuffer(Buffer):
    """A :class:`Buffer` at the address :meth:`MulticastObject.map` mapped.

    Every access through this address reaches the memory every device bound
    into the multicast object. Only the ``multimem`` PTX instructions
    (``multimem.ld_reduce``, ``multimem.st``, ``multimem.red``) have defined
    behavior on it. The buffer keeps the multicast object alive; closing it
    waits for its deallocation stream, then unmaps, as for
    :class:`VirtualMemoryBuffer`. It cannot be exported: share the
    :attr:`MulticastObject.ipc_descriptor` instead.
    """

    @property
    def ipc_descriptor(self):
        raise RuntimeError(
            "a MulticastBuffer cannot be exported; share MulticastObject.ipc_descriptor and map it in the "
            "receiving process"
        )

    def __reduce__(self):
        raise TypeError("a MulticastBuffer cannot be pickled; send MulticastObject.ipc_descriptor instead")

    def close(self, stream: Stream | GraphBuilder | None = None) -> None:
        """Release this buffer's mapping of the multicast object.

        Waits for the work queued on the deallocation stream, then unmaps.
        Refuses a capturing stream other than a default stream, because the
        unmap cannot be captured.

        Parameters
        ----------
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            If given, replaces the recorded deallocation stream, as for
            :meth:`Buffer.close`.
        """
        cdef Stream s
        cdef StreamHandle h
        if not self._h_ptr:
            return
        if stream is not None:
            s = Stream_accept(stream)
            h = s._h_stream
        else:
            h = deallocation_stream(self._h_ptr)
        if h and not Stream_handle_is_default_token(as_cu(h)) and _stream_is_capturing(as_cu(h)):
            raise RuntimeError(
                "cannot close a MulticastBuffer on a capturing stream: unmapping is synchronous and "
                "cannot be captured"
            )
        Buffer.close(self, stream)


cdef class MulticastObject:
    """A CUDA multicast object.

    Memory that several devices bind into one multicast object can be read
    with a reduction and written to all of them at once, through the address
    :meth:`map` returns, with the ``multimem`` PTX instructions. One process
    creates the object and shares :attr:`ipc_descriptor`; the others import
    it with :meth:`from_ipc_descriptor`. Then:

    1. every participant calls :meth:`add_device` for its devices;
    2. every participant agrees that all of them succeeded (for example with a
       collective of your own);
    3. every participant calls :meth:`bind` for its memory and :meth:`map`
       for its devices.

    :meth:`bind` and :meth:`map` block inside the driver until all
    ``num_devices`` devices have been added, with no timeout. A participant
    that fails to add its device and does not tell the others leaves them
    blocked; step 2 is what prevents it.

    The object, each :class:`MulticastBinding` and each
    :class:`MulticastBuffer` hold their own references, so they can be closed
    in any order; the multicast object is released with the last of them.

    Parameters
    ----------
    options : MulticastObjectOptions
        Number of devices, size, handle type and granularity.
    """

    cdef:
        MemAllocationHandle _h_mc
        size_t _size
        size_t _granularity
        int _num_devices
        object _handle_type
        object __weakref__

    def __init__(self, options: MulticastObjectOptions) -> None:
        cdef cydriver.CUmulticastObjectProp prop
        if not isinstance(options, MulticastObjectOptions):
            raise TypeError(f"options must be MulticastObjectOptions, got {type(options).__name__}")
        _require_multicast_driver()
        self._num_devices = options.num_devices
        self._handle_type = options.handle_type
        self._granularity = _query_granularity(options.num_devices, options.handle_type, options.granularity)
        self._size = (options.size + self._granularity - 1) // self._granularity * self._granularity
        _fill_prop(&prop, self._num_devices, self._size, self._handle_type)
        with nogil:
            self._h_mc = create_multicast_handle(prop)
        if not self._h_mc:
            _raise_last_error_noting(
                "while creating a multicast object; it needs devices with multicast support "
                "(Device.properties.multicast_supported), and fabric handles also need an accessible IMEX channel"
            )

    @staticmethod
    def get_granularity(
        num_devices: int,
        *,
        handle_type: VirtualMemoryHandleType | str | None = VirtualMemoryHandleType.FABRIC,
        granularity: VirtualMemoryGranularityType | str = VirtualMemoryGranularityType.RECOMMENDED,
    ) -> int:
        """Return the multicast granularity in bytes.

        Sizes and offsets passed to :meth:`bind` must be multiples of the
        minimum granularity, and memory bound with :meth:`bind` should be
        allocated in multiples of it.

        Parameters
        ----------
        num_devices : int
            Number of devices of the multicast object.
        handle_type : :obj:`~_memory.VirtualMemoryHandleType` | str | None, optional
            Handle type of the multicast object.
        granularity : :obj:`~_memory.VirtualMemoryGranularityType` | str, optional
            ``"recommended"`` (default) or ``"minimum"``.
        """
        return _query_granularity(
            _positive_int("num_devices", num_devices),
            _normalize_handle_type(handle_type),
            _normalize_granularity(granularity),
        )

    @classmethod
    def from_ipc_descriptor(cls, descriptor: MulticastObjectDescriptor) -> MulticastObject:
        """Import a multicast object another process exported.

        Parameters
        ----------
        descriptor : MulticastObjectDescriptor
            From :attr:`ipc_descriptor` in the exporting process.
        """
        cdef MulticastObject self
        cdef cydriver.CUmemFabricHandle fh
        cdef bytes raw
        cdef size_t size
        if not isinstance(descriptor, MulticastObjectDescriptor):
            raise TypeError(f"descriptor must be MulticastObjectDescriptor, got {type(descriptor).__name__}")
        _require_multicast_driver()
        raw = descriptor.handle
        size = descriptor.size
        memset(&fh, 0, sizeof(fh))
        memcpy(<void*>fh.data, <const char*>raw, _FABRIC_HANDLE_SIZE)
        self = cls.__new__(cls)
        self._num_devices = descriptor.num_devices
        self._handle_type = VirtualMemoryHandleType.FABRIC
        self._granularity = _query_granularity(
            self._num_devices, self._handle_type, VirtualMemoryGranularityType.RECOMMENDED)
        if size % _query_granularity(self._num_devices, self._handle_type, VirtualMemoryGranularityType.MINIMUM):
            raise ValueError(f"descriptor size {size} is not a multiple of the multicast granularity")
        with nogil:
            self._h_mc = import_mem_allocation_handle(
                <void*>&fh, cydriver.CU_MEM_HANDLE_TYPE_FABRIC, size, NULL, 0)
        if not self._h_mc:
            _raise_last_error_noting(
                "while importing a multicast object; the handle may be stale (the exporting process released "
                "the multicast object), forged, or exported on a different IMEX channel"
            )
        self._size = size
        return self

    @property
    def ipc_descriptor(self) -> MulticastObjectDescriptor:
        """A descriptor for sharing this multicast object with other processes.

        Requires ``handle_type="fabric"``. An imported multicast object cannot
        be exported again; forward the descriptor it was imported from.
        """
        cdef cydriver.CUmemFabricHandle fh
        self._check_open()
        if self._handle_type is None:
            raise RuntimeError("a multicast object created with handle_type=None cannot be exported")
        if mem_allocation_is_imported(self._h_mc):
            raise RuntimeError(
                "an imported multicast object cannot be exported again; forward the descriptor it was imported from"
            )
        memset(&fh, 0, sizeof(fh))
        with nogil:
            HANDLE_RETURN(cydriver.cuMemExportToShareableHandle(
                <void*>&fh, as_cu(self._h_mc), cydriver.CU_MEM_HANDLE_TYPE_FABRIC, 0))
        return MulticastObjectDescriptor(
            cpython.PyBytes_FromStringAndSize(<char*>fh.data, sizeof(fh.data)), self._size, self._num_devices)

    @property
    def size(self) -> int:
        """Size of the multicast object in bytes, rounded to the granularity."""
        return self._size

    @property
    def num_devices(self) -> int:
        """Number of devices the multicast object was created for."""
        return self._num_devices

    @property
    def granularity(self) -> int:
        """The granularity the size was rounded to, and the alignment :meth:`map` reserves with."""
        return self._granularity

    @property
    def handle_type(self) -> VirtualMemoryHandleType | None:
        """The handle type the multicast object was created with."""
        return self._handle_type

    @property
    def handle(self) -> int:
        """The ``CUmemGenericAllocationHandle`` of the multicast object, as an int."""
        self._check_open()
        return <unsigned long long>as_cu(self._h_mc)

    @property
    def is_closed(self) -> bool:
        """Whether this object was closed."""
        return not self._h_mc

    cdef int _check_open(self) except -1:
        if not self._h_mc:
            raise RuntimeError("MulticastObject has been closed")
        return 0

    def add_device(self, device: Device | int) -> None:
        """Add a device to the multicast object (``cuMulticastAddDevice``).

        Does not block. Every participant must add its devices before any
        participant can bind memory.

        Parameters
        ----------
        device : :obj:`~_device.Device` | int
            The device to add.
        """
        cdef int ordinal = _device_ordinal(device)
        cdef cydriver.CUdevice cu_dev
        self._check_open()
        if not Device(ordinal).properties.multicast_supported:
            raise RuntimeError(f"device {ordinal} does not support multicast objects")
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGet(&cu_dev, ordinal))
            HANDLE_RETURN(cydriver.cuMulticastAddDevice(as_cu(self._h_mc), cu_dev))

    def bind(self, buffer not None: VirtualMemoryBuffer, *, offset: int = 0, size: int | None = None) -> MulticastBinding:
        """Bind the physical memory of a :class:`VirtualMemoryBuffer` (``cuMulticastBindMem``).

        Every physical allocation that backs the first ``size`` bytes of
        ``buffer`` is bound, in address order, starting at ``offset`` in the
        multicast object. Blocks until every device of the multicast object
        has been added (see the class notes).

        Parameters
        ----------
        buffer : VirtualMemoryBuffer
            Device memory this process allocated with a
            :class:`VirtualMemoryResource`. Imported memory cannot be bound.
        offset : int, optional
            Offset in the multicast object, a multiple of the minimum
            granularity.
        size : int, optional
            Bytes to bind, a multiple of the minimum granularity. Defaults to
            ``buffer.size``.

        Returns
        -------
        MulticastBinding
            Unbinds the memory when closed.
        """
        cdef Buffer buf
        cdef DevicePtrHandle h_ptr
        cdef vector[VaMappingHandle] mappings
        cdef vector[MulticastBindingHandle] bindings
        cdef MulticastBindingHandle h_bind
        cdef MemAllocationHandle h_alloc
        cdef MemAllocationHandle h_mc = self._h_mc
        cdef cydriver.CUdevice cu_dev
        cdef size_t gran_min, want, done = 0, chunk, mc_offset
        cdef size_t i, n
        cdef int ordinal
        cdef MulticastBinding result

        self._check_open()
        if not isinstance(buffer, VirtualMemoryBuffer):
            raise TypeError(f"buffer must be a VirtualMemoryBuffer, got {type(buffer).__name__}")
        buf = <Buffer>buffer
        Buffer_check_open(buf)
        mr = buf._memory_resource
        if mr.device is None or mr.config.location_type != VirtualMemoryLocationType.DEVICE:
            raise ValueError("only device memory can be bound into a multicast object")
        ordinal = mr.device.device_id
        gran_min = _query_granularity(self._num_devices, self._handle_type, VirtualMemoryGranularityType.MINIMUM)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"offset must be a non-negative int, got {offset!r}")
        want = buf._size if size is None else _positive_int("size", size)
        if want == 0:
            raise ValueError("cannot bind an empty buffer")
        if want > buf._size:
            raise ValueError(f"size {want} exceeds the buffer's {buf._size} bytes")
        if offset % gran_min or want % gran_min:
            raise ValueError(
                f"offset ({offset}) and size ({want}) must be multiples of the minimum multicast granularity "
                f"({gran_min} bytes)"
            )
        if offset + want > self._size:
            raise ValueError(f"offset {offset} + size {want} exceeds the multicast object's {self._size} bytes")

        h_ptr = buf._h_ptr
        mappings = vmm_range_mappings(vmm_range(h_ptr))
        n = mappings.size()
        for i in range(n):
            if done >= want:
                break
            if mem_allocation_is_imported(va_mapping_allocation(mappings[i])):
                raise ValueError("memory imported from another process cannot be bound into a multicast object")
            done += mem_allocation_size(va_mapping_allocation(mappings[i]))

        mc_offset = offset
        done = 0
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGet(&cu_dev, ordinal))
        for i in range(n):
            if done >= want:
                break
            h_alloc = va_mapping_allocation(mappings[i])
            chunk = min(mem_allocation_size(h_alloc), want - done)
            with nogil:
                h_bind = create_multicast_binding_handle(h_mc, mc_offset + done, h_alloc, 0, chunk, cu_dev)
            if not h_bind:
                _raise_last_error_noting(
                    f"while binding allocation {i + 1} of the buffer at multicast offset {mc_offset + done}"
                )
            bindings.push_back(h_bind)
            done += chunk

        result = MulticastBinding.__new__(MulticastBinding)
        result._bindings.swap(bindings)
        result._offset = offset
        result._size = want
        result._device_id = ordinal
        return result

    def map(self, device: Device | int, *, stream: Stream | GraphBuilder | None = None) -> MulticastBuffer:
        """Map the multicast object for ``device`` and return a buffer at its address.

        Blocks until every device of the multicast object has been added.

        Parameters
        ----------
        device : :obj:`~_device.Device` | int
            The device that gets read-write access to the mapping.
        stream : :obj:`~_stream.Stream` | :obj:`~graph.GraphBuilder`, optional
            Recorded as the buffer's deallocation stream and synchronized
            when it closes. The legacy default stream of ``device`` is
            recorded by default.
        """
        cdef int ordinal = _device_ordinal(device)
        cdef cydriver.CUmemAccessDesc desc
        cdef VaReservationHandle h_res
        cdef VaMappingHandle h_map
        cdef vector[VaMappingHandle] mappings
        cdef VmmRangeHandle rng
        cdef DevicePtrHandle h_ptr
        cdef MemAllocationHandle h_mc = self._h_mc
        cdef StreamHandle h_stream
        cdef ContextHandle h_ctx
        cdef Stream s = None
        cdef size_t size = self._size, gran = self._granularity

        self._check_open()
        if stream is not None:
            s = Stream_accept(stream)
        memset(&desc, 0, sizeof(desc))
        desc.location.type = cydriver.CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = ordinal
        desc.flags = cydriver.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        with nogil:
            h_res = create_va_reservation_handle(size, gran, 0)
        if not h_res:
            _raise_last_error()
        with nogil:
            h_map = create_va_mapping_handle_with_access(as_cu(h_res), h_mc, h_res, &desc, 1)
        if not h_map:
            _raise_last_error_noting(f"while mapping the multicast object for device {ordinal}")
        mappings.push_back(h_map)
        rng = create_vmm_range(mappings)
        h_ptr = deviceptr_create_vmm(as_cu(h_res), rng)
        if not h_ptr:
            _raise_last_error()
        if s is not None and not Stream_is_default_token(s):
            h_stream = s._h_stream
        else:
            h_ctx = get_primary_context(ordinal)
            if not h_ctx:
                _raise_last_error()
            h_stream = create_context_bound_legacy_stream(h_ctx)
            if not h_stream:
                _raise_last_error()
        HANDLE_RETURN(set_deallocation_stream(h_ptr, h_stream))
        return Buffer_from_deviceptr_handle(h_ptr, size, None, None, MulticastBuffer)

    def close(self) -> None:
        """Release this object's reference to the multicast object.

        Bindings and mapped buffers keep their own references; the multicast
        object is released with the last of them. Calling it again does
        nothing.
        """
        self._h_mc.reset()

    def __enter__(self) -> MulticastObject:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __reduce__(self):
        raise TypeError("a MulticastObject cannot be pickled; send its ipc_descriptor instead")

    def __repr__(self) -> str:
        if not self._h_mc:
            return "<MulticastObject closed>"
        return (
            f"<MulticastObject size={self._size} num_devices={self._num_devices} "
            f"handle_type={self._handle_type!s}>"
        )
