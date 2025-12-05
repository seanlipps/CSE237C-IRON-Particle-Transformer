
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

# score 
@iron.jit(is_placed=False)
def score_ly(input0, input1, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, depth=1, name="x")
    of_y = ObjectFifo(in_ty, depth=1, name="y")
    of_z = ObjectFifo(out_ty, depth=1, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    score_ly_kernel = ExternalFunction(
        "score_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/score_layer.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_y, of_z, score_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        score_ly_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), score_ly_kernel]
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
    
    inp = np.loadtxt("./score_data/input.txt", dtype=np.int8)
    ref = np.loadtxt("./data/a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 160
    INPUT_COLS = 16

    inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input
    inp_tensor0 = iron.tensor(inp_tiled, dtype=np.int8, device="npu")
    inp_tensor1 = iron.tensor(inp_tiled, dtype=np.int8, device="npu")
    score_output = iron.zeros(160*160, dtype=element_type, device="npu")

    # Insantiate AIE Kernel           
    score_ly(inp_tensor0, inp_tensor1, score_output) # 160*16 @ 160*16^T = 160*16 @ 16*160 = 160*160
    
    out_np = np.array(score_output, dtype=np.int8)

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
