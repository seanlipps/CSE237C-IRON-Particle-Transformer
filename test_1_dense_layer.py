
# from ml_dtypes import bfloat16
import numpy as np
import sys
import os
from utils.tiling import tile_matrix
import time
import argparse

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.config import cxx_header_path


# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - is_placed (bool): Whether the kernel is using explicit or deferred placement API. Defaults to True.
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit(is_placed=False)
def dense_ly(input0, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, name="x")
    of_z = ObjectFifo(out_ty, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel acquires input tensors X and Y, and output tensor Z, performs the
    # SAXPY operation on X and Y, and writes the result in Z.

    dense_ly_kernel = ExternalFunction(
        "f0",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_1_layer_0.cc"),
        arg_types=[in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_z, dense_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_z = of_z.acquire(1)
        dense_ly_kernel(elem_x, elem_z)
        of_x.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_z.prod(), dense_ly_kernel]
    )

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_x, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.drain(of_z.cons(), c_z, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())


def main():
    argparser = argparse.ArgumentParser(
            prog="Dense Kernel Test",
            description="Programming testing if dense layer works"
            )
    argparser.add_argument('-b', '--benchmark', action='store_true', help=argparse.SUPPRESS)
    args = argparser.parse_args()


    element_type = np.int8
    
    inp = np.loadtxt("./iron_kernels/test_data/test_1_dense_input.txt", dtype=np.int8)
    ref = np.loadtxt("./iron_kernels/test_data/test_1_dense_out_ref.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 160
    INPUT_COLS = 8
    OUTPUT_SIZE = 160 * 64

    if inp.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp.size} != {INPUT_ROWS*INPUT_COLS}")

    inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input and output
    inp_tensor = iron.zeros(inp_tiled.shape, dtype=np.int8, device="npu")
    inp_tensor[:] = inp_tiled
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    dense_ly(inp_tensor, output)

    # Measure peformance on the second execution using the JIT cached design
    # Optional to run the test
    if args.benchmark:
        output_ben = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")
        inp_tensor_ben    = iron.zeros(inp_tiled.shape, dtype=np.int8, device="npu")
        inp_tensor_ben[:] = inp_tiled

        # benchmark performance. 
        # Will use jit compiled kernel and loaded objects
        start_time = time.perf_counter()
        end_time = time.perf_counter()

        # benchark
        elapsed_time = end_time - start_time  # seconds
        dense_ly(inp_tensor_ben, output_ben)
        elapsed_us = elapsed_time * 1e6  # microseconds

        # Bandwidth calculation
        #total_bytes = 2.0 * length * np.dtype(element_type).itemsize  # input + output
        #bandwidth_GBps = total_bytes / elapsed_us / 1e3  # (bytes / µs) → GB/s

        print(f"Latency: {elapsed_time:.6f} seconds ({elapsed_us:.2f} µs)")
        #print(f"Effective Bandwidth: {bandwidth_GBps:.2f} GB/s")


    out_np = np.array(output, dtype=np.int8)

    errors = 0
    for i, (a, r) in enumerate(zip(out_np, ref)):
        if a != r:
            print(f"Error at {i}: {a} != {r}")
            errors += 1

    if errors == 0:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print(f"\nError count: {errors}")
        print("failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
