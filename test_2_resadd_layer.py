
# from ml_dtypes import bfloat16
import numpy as np
import sys
import os
from utils.tiling import tile_matrix

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
def resadd_ly(input0, input1, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel acquires input tensors X and Y, and output tensor Z, performs the
    # SAXPY operation on X and Y, and writes the result in Z.

    resadd_ly_kernel = ExternalFunction(
        "f0",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_2_layer_0.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_y, of_z, resadd_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        resadd_ly_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), resadd_ly_kernel]
    )

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())


def main():
    element_type = np.int8
    
    inp0 = np.loadtxt("./iron_kernels/test_data/test_2_input0.txt", dtype=np.int8)
    inp1 = np.loadtxt("./iron_kernels/test_data/test_2_input1.txt", dtype=np.int8)
    ref = np.loadtxt("./iron_kernels/test_data/test_2_out_ref.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 160
    INPUT_COLS = 8
    OUTPUT_SIZE = 160 * 8

    if inp0.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input0 size {inp.size} != {INPUT_ROWS*INPUT_COLS}")
    if inp1.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input1 size {inp.size} != {INPUT_ROWS*INPUT_COLS}")

    inp0_mat = inp0.reshape(INPUT_ROWS, INPUT_COLS)
    inp0_tiled = tile_matrix(inp0_mat, 4, 8)  # flattened tiled input
    inp1_mat = inp1.reshape(INPUT_ROWS, INPUT_COLS)
    inp1_tiled = tile_matrix(inp1_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input and output
    inp0_tensor = iron.tensor(inp0_tiled, dtype=np.int8, device="npu")
    inp1_tensor = iron.tensor(inp1_tiled, dtype=np.int8, device="npu")
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    resadd_ly(inp0_tensor, inp1_tensor, output)

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
