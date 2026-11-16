# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multicast objects bound to VirtualMemoryResource buffers (issue #2980)."""

import ctypes
import multiprocessing as mp
import pickle

import numpy as np
import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses

from cuda.bindings import driver
from cuda.core import (
    Buffer,
    Device,
    LaunchConfig,
    MulticastBinding,
    MulticastBuffer,
    MulticastObject,
    MulticastObjectDescriptor,
    MulticastObjectOptions,
    Program,
    ProgramOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
    launch,
)
from cuda.core._utils.cuda_utils import CUDAError, handle_return
from cuda.core.typing import VirtualMemoryGranularityType, VirtualMemoryHandleType

CHILD_TIMEOUT_SEC = child_timeout_sec()
pytestmark = pytest.mark.parallel_threads_limit(1)

_KERNELS = r"""
extern "C" __global__ void store(float* mc, float value, unsigned long long n) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" :: "l"(mc + i), "f"(value) : "memory");
  }
}

extern "C" __global__ void reduce(const float* mc, float* out, unsigned long long n) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v;
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(v) : "l"(mc + i) : "memory");
    out[i] = v;
  }
}
"""

_N = 4096


def _kernels(device):
    opts = ProgramOptions(arch=f"sm_{device.arch}")
    return Program(_KERNELS, "c++", options=opts).compile("cubin")


def _store(device, mbuf, value, n=_N):
    stream = device.create_stream()
    launch(
        stream,
        LaunchConfig(grid=n // 256, block=256),
        _kernels(device).get_kernel("store"),
        mbuf,
        np.float32(value),
        np.uint64(n),
    )
    stream.sync()
    stream.close()


def _reduce(device, mbuf, n=_N):
    stream = device.create_stream()
    with device.allocate(n * 4, stream=stream) as out:
        launch(
            stream,
            LaunchConfig(grid=n // 256, block=256),
            _kernels(device).get_kernel("reduce"),
            mbuf,
            out,
            np.uint64(n),
        )
        stream.sync()
        values = _read(out, n)
    stream.close()
    return values


def _read(buf, n=_N):
    host = (ctypes.c_float * n)()
    handle_return(driver.cuMemcpyDtoH(ctypes.addressof(host), int(buf.handle), n * 4))
    return np.frombuffer(host, dtype=np.float32).copy()


def _fill(buf, value):
    handle_return(driver.cuMemsetD32(int(buf.handle), int(np.float32(value).view(np.uint32)), buf.size // 4))
    handle_return(driver.cuCtxSynchronize())


def _vmm_buffer(device, size, handle_type=None):
    mr = VirtualMemoryResource(device, VirtualMemoryResourceOptions(handle_type=handle_type))
    return mr.allocate(size)


@pytest.fixture
def multicast_device_x2(device_x2):
    for device in device_x2:
        if not device.properties.multicast_supported:
            pytest.skip("Multicast objects are not supported on this device")
        if not device.properties.virtual_memory_management_supported:
            pytest.skip("Virtual memory management is not supported on this device")
    return device_x2


@pytest.fixture
def multicast_device(init_cuda):
    if not init_cuda.properties.multicast_supported:
        pytest.skip("Multicast objects are not supported on this device")
    return init_cuda


def test_options_validation():
    opts = MulticastObjectOptions(num_devices=2, size=1, handle_type="fabric", granularity="minimum")
    assert opts.handle_type is VirtualMemoryHandleType.FABRIC
    assert opts.granularity is VirtualMemoryGranularityType.MINIMUM
    assert MulticastObjectOptions(num_devices=1, size=1, handle_type=None).handle_type is None
    for bad in ({"num_devices": 0}, {"num_devices": True}, {"size": 0}, {"size": 1.5}):
        with pytest.raises(ValueError):
            MulticastObjectOptions(**{"num_devices": 2, "size": 1, **bad})
    with pytest.raises(ValueError, match="not supported for multicast"):
        MulticastObjectOptions(num_devices=2, size=1, handle_type="posix_fd")
    with pytest.raises(ValueError, match="invalid handle_type"):
        MulticastObjectOptions(num_devices=2, size=1, handle_type="nope")
    with pytest.raises(ValueError, match="invalid granularity"):
        MulticastObjectOptions(num_devices=2, size=1, granularity="nope")


def test_descriptor_pickle_and_validation():
    desc = MulticastObjectDescriptor(bytes(range(64)), 2 << 20, 2)
    again = pickle.loads(pickle.dumps(desc))  # noqa: S301  our own plain data
    assert again == desc and hash(again) == hash(desc)
    assert (again.handle, again.size, again.num_devices) == (bytes(range(64)), 2 << 20, 2)
    assert again.handle_type is VirtualMemoryHandleType.FABRIC
    assert desc != MulticastObjectDescriptor(bytes(64), 2 << 20, 2)
    with pytest.raises(ValueError, match="64 bytes"):
        MulticastObjectDescriptor(bytes(63), 2 << 20, 2)
    with pytest.raises(ValueError, match="size"):
        MulticastObjectDescriptor(bytes(64), 0, 2)
    with pytest.raises(ValueError, match="num_devices"):
        MulticastObjectDescriptor(bytes(64), 2 << 20, 0)


def test_granularity(multicast_device):
    minimum = MulticastObject.get_granularity(2, handle_type=None, granularity="minimum")
    recommended = MulticastObject.get_granularity(2, handle_type=None)
    assert minimum > 0 and recommended >= minimum and recommended % minimum == 0
    with pytest.raises(ValueError):
        MulticastObject.get_granularity(0)


def test_size_rounds_to_granularity(multicast_device):
    with MulticastObject(MulticastObjectOptions(num_devices=2, size=1, handle_type=None)) as mc:
        assert mc.size == mc.granularity == MulticastObject.get_granularity(2, handle_type=None)
        assert mc.num_devices == 2 and mc.handle_type is None and mc.handle != 0
        assert "num_devices=2" in repr(mc)
    assert mc.is_closed and "closed" in repr(mc)
    mc.close()
    with pytest.raises(RuntimeError, match="closed"):
        mc.handle  # noqa: B018


def test_store_and_reduce_in_one_process(multicast_device_x2):
    dev0, dev1 = multicast_device_x2
    mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=1 << 20, handle_type=None))
    mc.add_device(dev0)
    mc.add_device(dev1)
    bufs = [_vmm_buffer(dev, mc.size) for dev in (dev0, dev1)]
    bindings = [mc.bind(buf) for buf in bufs]
    assert [b.device_id for b in bindings] == [dev0.device_id, dev1.device_id]
    assert all(isinstance(b, MulticastBinding) and b.offset == 0 and b.size == mc.size for b in bindings)
    dev0.set_current()
    mbuf = mc.map(dev0)
    assert isinstance(mbuf, MulticastBuffer) and mbuf.size == mc.size and mbuf.memory_resource is None

    _store(dev0, mbuf, 1.5)
    assert np.all(_read(bufs[0]) == 1.5)
    dev1.set_current()
    assert np.all(_read(bufs[1]) == 1.5)
    dev0.set_current()

    _fill(bufs[0], 2.0)
    dev1.set_current()
    _fill(bufs[1], 3.0)
    dev0.set_current()
    assert np.all(_reduce(dev0, mbuf) == 5.0)

    # The object's own reference goes first; the binding and the mapping keep the multicast object alive.
    mc.close()
    _store(dev0, mbuf, 4.0)
    assert np.all(_read(bufs[0]) == 4.0)
    mbuf.close()
    for binding in bindings:
        binding.close()
        assert binding.is_closed
        binding.close()
    for buf in bufs:
        buf.close()


def test_release_in_any_order(multicast_device_x2):
    dev0, dev1 = multicast_device_x2
    for order in ("buffers_first", "bindings_first"):
        mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=1, handle_type=None))
        mc.add_device(dev0)
        mc.add_device(dev1)
        bufs = [_vmm_buffer(dev, mc.size) for dev in (dev0, dev1)]
        bindings = [mc.bind(buf) for buf in bufs]
        dev0.set_current()
        mbuf = mc.map(dev0)
        if order == "buffers_first":
            for buf in bufs:
                buf.close()
            mc.close()
            mbuf.close()
            del bindings
        else:
            del bindings
            mbuf.close()
            mc.close()
            for buf in bufs:
                buf.close()


def test_bind_grown_buffer_binds_every_chunk(multicast_device_x2):
    dev0, dev1 = multicast_device_x2
    gran = MulticastObject.get_granularity(2, handle_type=None)
    mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=2 * gran, handle_type=None))
    mc.add_device(dev0)
    mc.add_device(dev1)
    mr0 = VirtualMemoryResource(dev0, VirtualMemoryResourceOptions(handle_type=None))
    small = mr0.allocate(gran)
    grown = mr0.modify_allocation(small, 2 * gran)
    other = _vmm_buffer(dev1, 2 * gran)
    with mc.bind(grown) as b0, mc.bind(other) as b1:
        assert b0.size == b1.size == 2 * gran
        dev0.set_current()
        with mc.map(dev0) as mbuf:
            n = (2 * gran) // 4
            _store(dev0, mbuf, 6.0, n=n)
            assert np.all(_read(grown, n) == 6.0)
    for buf in (small, grown, other):
        buf.close()


def test_bind_validation(multicast_device_x2):
    dev0, dev1 = multicast_device_x2
    gran = MulticastObject.get_granularity(2, handle_type=None, granularity="minimum")
    mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=2 * gran, handle_type=None))
    mc.add_device(dev0)
    mc.add_device(dev1)
    with _vmm_buffer(dev0, 4 * gran) as big, dev0.allocate(gran, stream=dev0.default_stream) as plain:
        with pytest.raises(TypeError, match="VirtualMemoryBuffer"):
            mc.bind(plain)
        with pytest.raises(ValueError, match="exceeds the multicast"):
            mc.bind(big)
        with pytest.raises(ValueError, match="multiples of the minimum"):
            mc.bind(big, size=gran, offset=gran // 2)
        with pytest.raises(ValueError, match="exceeds the buffer"):
            mc.bind(big, size=8 * gran)
        with pytest.raises(ValueError, match="non-negative"):
            mc.bind(big, size=gran, offset=-gran)
        with mc.bind(big, size=gran, offset=gran) as binding:
            assert (binding.offset, binding.size) == (gran, gran)
    mc.close()
    with pytest.raises(RuntimeError, match="closed"):
        mc.map(dev0)


def test_not_exportable_or_picklable(multicast_device_x2):
    dev0, dev1 = multicast_device_x2
    mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=1, handle_type=None))
    with pytest.raises(RuntimeError, match="handle_type=None"):
        mc.ipc_descriptor  # noqa: B018
    with pytest.raises(TypeError):
        pickle.dumps(mc)
    mc.add_device(dev0)
    mc.add_device(dev1)
    with _vmm_buffer(dev0, mc.size) as b0, _vmm_buffer(dev1, mc.size) as b1, mc.bind(b0) as binding, mc.bind(b1):
        with pytest.raises(TypeError):
            pickle.dumps(binding)
        with pytest.raises(RuntimeError, match="created by MulticastObject.bind"):
            MulticastBinding()
        dev0.set_current()
        with mc.map(dev0) as mbuf:
            with pytest.raises(RuntimeError, match="cannot be exported"):
                mbuf.ipc_descriptor  # noqa: B018
            with pytest.raises(TypeError):
                pickle.dumps(mbuf)
    mc.close()


def _require_fabric(device):
    if not device.properties.handle_type_fabric_supported:
        pytest.skip("Fabric handles need an NVLink fabric and an accessible IMEX channel")


def test_imported_memory_cannot_be_bound(vmm_fabric_device_x2):
    dev0, dev1 = vmm_fabric_device_x2
    for dev in (dev0, dev1):
        if not dev.properties.multicast_supported:
            pytest.skip("Multicast objects are not supported on this device")
    mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=1))
    with _vmm_buffer(dev0, mc.size, "fabric") as owned:
        mr = VirtualMemoryResource(dev0, VirtualMemoryResourceOptions(handle_type="fabric"))
        alias = Buffer.from_ipc_descriptor(mr, owned.ipc_descriptor, stream=dev0.default_stream)
        with alias, pytest.raises(ValueError, match="imported from another process"):
            mc.bind(alias)
    mc.close()


def test_fabric_descriptor_shape_and_reexport(vmm_fabric_device_x2):
    dev0, _ = vmm_fabric_device_x2
    if not dev0.properties.multicast_supported:
        pytest.skip("Multicast objects are not supported on this device")
    with MulticastObject(MulticastObjectOptions(num_devices=2, size=1)) as mc:
        desc = mc.ipc_descriptor
        assert len(desc.handle) == 64 and desc.size == mc.size and desc.num_devices == 2
        assert mc.ipc_descriptor == desc
        rebuilt = pickle.loads(pickle.dumps(desc))  # noqa: S301  our own plain data
        with MulticastObject.from_ipc_descriptor(rebuilt) as imported:
            assert imported.size == mc.size and imported.handle_type is VirtualMemoryHandleType.FABRIC
            with pytest.raises(RuntimeError, match="cannot be exported again"):
                imported.ipc_descriptor  # noqa: B018
    with pytest.raises(TypeError):
        MulticastObject.from_ipc_descriptor(desc.handle)


def test_garbage_descriptor(vmm_fabric_device_x2):
    dev0, _ = vmm_fabric_device_x2
    if not dev0.properties.multicast_supported:
        pytest.skip("Multicast objects are not supported on this device")
    gran = MulticastObject.get_granularity(2)
    with pytest.raises(CUDAError) as excinfo:
        MulticastObject.from_ipc_descriptor(MulticastObjectDescriptor(b"\xa5" * 64, gran, 2))
    assert "stale" in " ".join(excinfo.value.__notes__)


def _child_join(desc, device_id, conn):
    device = Device(device_id)
    device.set_current()
    mc = MulticastObject.from_ipc_descriptor(desc)
    mc.add_device(device)
    buf = _vmm_buffer(device, mc.size, "fabric")
    binding = mc.bind(buf)
    mbuf = mc.map(device)
    conn.send("joined")
    assert conn.poll(CHILD_TIMEOUT_SEC)
    conn.recv()
    conn.send(float(np.unique(_read(buf))[0]))
    _store(device, mbuf, 9.0)
    conn.send("stored")
    assert conn.poll(CHILD_TIMEOUT_SEC)
    conn.recv()
    mbuf.close()
    binding.close()
    buf.close()
    mc.close()
    conn.close()


def test_cross_process(vmm_fabric_device_x2):
    dev0, dev1 = vmm_fabric_device_x2
    for dev in (dev0, dev1):
        if not dev.properties.multicast_supported:
            pytest.skip("Multicast objects are not supported on this device")
    mc = MulticastObject(MulticastObjectOptions(num_devices=2, size=1))
    parent, child = mp.Pipe()
    proc = mp.get_context("spawn").Process(target=_child_join, args=(mc.ipc_descriptor, dev1.device_id, child))
    proc.start()
    try:
        mc.add_device(dev0)
        buf = _vmm_buffer(dev0, mc.size, "fabric")
        binding = mc.bind(buf)
        dev0.set_current()
        mbuf = mc.map(dev0)
        assert parent.poll(CHILD_TIMEOUT_SEC) and parent.recv() == "joined"
        _store(dev0, mbuf, 7.25)
        parent.send("go")
        assert parent.poll(CHILD_TIMEOUT_SEC) and parent.recv() == 7.25
        assert parent.poll(CHILD_TIMEOUT_SEC) and parent.recv() == "stored"
        assert np.all(_read(buf) == 9.0)
        parent.send("done")
        proc.join(CHILD_TIMEOUT_SEC)
        assert proc.exitcode == 0
        mbuf.close()
        binding.close()
        buf.close()
        mc.close()
    finally:
        kill_subprocesses(proc)
