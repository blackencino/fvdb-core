# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
CUDA kernel JIT compilation and launch via cuda-python.

Provides a thin utility layer over ``cuda.bindings.nvrtc`` (JIT compile
CUDA C++ to PTX/CUBIN) and ``cuda.bindings.driver`` (load module, launch
kernel on a CUDA stream).

This is a general-purpose utility -- not hash-map-specific.  Any CUDA
kernel source can be compiled and launched through these functions.

Compiled modules are cached by a hash of the source string so that
repeated calls with the same source avoid recompilation.

Usage::

    from cuda_launch import compile_and_get_function, launch_kernel
    import torch

    func = compile_and_get_function(CUDA_SOURCE, "my_kernel")
    t_in = torch.zeros(1024, dtype=torch.int64, device="cuda")
    t_out = torch.zeros(1024, dtype=torch.int64, device="cuda")
    launch_kernel(func, grid=(4,), block=(256,),
                  args=[t_in, t_out, 1024])
"""

from __future__ import annotations

import ctypes
import hashlib
from typing import Any

import numpy as np
import torch

from cuda.bindings import driver, nvrtc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_nvrtc(result):
    """Check NVRTC call result, raise on error."""
    err = result[0] if isinstance(result, tuple) else result
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, err_str = nvrtc.nvrtcGetErrorString(err)
        raise RuntimeError(f"NVRTC error: {err_str}")
    return result


def _check_driver(result):
    """Check driver API call result, raise on error."""
    err = result[0] if isinstance(result, tuple) else result
    if err != driver.CUresult.CUDA_SUCCESS:
        _, err_str = driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA driver error: {err_str}")
    return result


def _get_nvrtc_log(prog) -> str:
    """Extract the NVRTC compilation log from a program."""
    try:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        if err == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 0:
            log = b"\0" * log_size
            nvrtc.nvrtcGetProgramLog(prog, log)
            return log.decode(errors="replace").rstrip("\0")
    except Exception:
        pass
    return "(could not retrieve compilation log)"


_CONTEXT_INITIALIZED = False


def _ensure_cuda_context():
    """Ensure PyTorch's CUDA context is initialized and active for driver API calls."""
    global _CONTEXT_INITIALIZED
    if not _CONTEXT_INITIALIZED:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available")
        # Force PyTorch to initialize its CUDA context
        torch.cuda.init()
        # A small allocation ensures the context is fully active
        torch.zeros(1, device="cuda")
        # Also initialize the driver API so it sees the context
        err = driver.cuInit(0)
        if isinstance(err, tuple):
            err = err[0]
        # Push PyTorch's context as current for driver API
        err, ctx = driver.cuCtxGetCurrent()
        if isinstance(err, int) or err != driver.CUresult.CUDA_SUCCESS or not ctx:
            # If no context is current, get the primary context for device 0
            err, device = driver.cuDeviceGet(0)
            err, ctx = driver.cuDevicePrimaryCtxRetain(device)
            driver.cuCtxPushCurrent(ctx)
        _CONTEXT_INITIALIZED = True


def _get_arch_flag() -> str:
    """Get the compute architecture flag for the current device."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    major, minor = torch.cuda.get_device_capability()
    return f"--gpu-architecture=compute_{major}{minor}"


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------

_MODULE_CACHE: dict[str, Any] = {}


def compile_and_get_function(source: str, kernel_name: str) -> Any:
    """JIT compile CUDA C++ source and return a CUfunction handle.

    Results are cached by (source_hash, kernel_name).

    Args:
        source: CUDA C++ kernel source string.
        kernel_name: name of the ``__global__`` function to extract.

    Returns:
        A ``CUfunction`` handle suitable for ``cuLaunchKernel``.
    """
    _ensure_cuda_context()

    cache_key = hashlib.sha256(source.encode()).hexdigest()[:16] + f"_{kernel_name}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    arch_flag = _get_arch_flag()

    # Create NVRTC program
    err, prog = nvrtc.nvrtcCreateProgram(
        source.encode(), b"hashmap.cu", 0, [], []
    )
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {err}")

    # Compile to PTX
    options = [arch_flag.encode()]
    err = nvrtc.nvrtcCompileProgram(prog, len(options), options)
    if isinstance(err, tuple):
        err = err[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_str = _get_nvrtc_log(prog)
        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(f"NVRTC compilation failed:\n{log_str}")

    # Get PTX (requires pre-allocated buffer)
    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcGetPTXSize failed: {err}")
    ptx = b"\0" * ptx_size
    err = nvrtc.nvrtcGetPTX(prog, ptx)
    if isinstance(err, tuple):
        err = err[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcGetPTX failed: {err}")
    nvrtc.nvrtcDestroyProgram(prog)

    # Load module from PTX
    err, module = driver.cuModuleLoadData(ptx)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleLoadData failed: {err}")

    # Get kernel function
    err, func = driver.cuModuleGetFunction(module, kernel_name.encode())
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction failed: {err}")

    _MODULE_CACHE[cache_key] = func
    return func


# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------


def _pack_arg(arg) -> np.ndarray:
    """Pack a single kernel argument into a numpy array for cuLaunchKernel.

    Supports:
      - torch.Tensor: uses data_ptr() as a device pointer
      - int: packed as uint64 (for size_t / pointer-width scalars)
      - ctypes pointer: packed as uint64
    """
    if isinstance(arg, torch.Tensor):
        return np.array([arg.data_ptr()], dtype=np.uint64)
    elif isinstance(arg, int):
        return np.array([arg], dtype=np.uint64)
    elif isinstance(arg, ctypes.c_void_p):
        return np.array([arg.value or 0], dtype=np.uint64)
    else:
        raise TypeError(f"Unsupported kernel arg type: {type(arg)}")


def launch_kernel(
    func,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: list,
    stream=None,
    shared_mem: int = 0,
):
    """Launch a CUDA kernel on a torch CUDA stream.

    Args:
        func: CUfunction handle from compile_and_get_function.
        grid: (gridDimX,) or (gridDimX, gridDimY) or (gridDimX, gridDimY, gridDimZ).
        block: (blockDimX,) or (blockDimX, blockDimY) or (blockDimX, blockDimZ, blockDimZ).
        args: list of kernel arguments (torch.Tensor, int, or ctypes pointers).
        stream: CUDA stream.  If None, uses torch.cuda.current_stream().
        shared_mem: dynamic shared memory size in bytes.
    """
    # Pad grid/block to 3D
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1
    bx = block[0] if len(block) > 0 else 1
    by = block[1] if len(block) > 1 else 1
    bz = block[2] if len(block) > 2 else 1

    # Pack arguments: each arg becomes a numpy array, then we build
    # the void** array of pointers to each arg's storage
    packed_args = [_pack_arg(a) for a in args]
    arg_ptrs = np.array([a.ctypes.data for a in packed_args], dtype=np.uint64)

    # Get stream handle
    if stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    elif isinstance(stream, int):
        cuda_stream = stream
    else:
        cuda_stream = stream.cuda_stream

    # Convert stream handle to CUstream
    cu_stream = driver.CUstream(cuda_stream)

    err = driver.cuLaunchKernel(
        func,
        gx, gy, gz,
        bx, by, bz,
        shared_mem,
        cu_stream,
        arg_ptrs,
        0,  # extra (unused)
    )
    if isinstance(err, tuple):
        err = err[0]
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {err}")
