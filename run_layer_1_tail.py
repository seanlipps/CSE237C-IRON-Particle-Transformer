import numpy as np
import sys
import os
from utils.tiling import tile_matrix

import aie.iron as iron
from aie.iron import ExternalFunction
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.config import cxx_header_path
from aie.dialects.aie import *  # primary mlir-aie dialect definitions


########################################
@iron.jit(is_placed=False, use_cache=False)
def mha_tail(input0, input1, input2, input3, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_0 = ObjectFifo(in_ty, name="head_0_in")
    of_1 = ObjectFifo(in_ty, name="head_1_in")
    of_2 = ObjectFifo(in_ty, name="head_2_in")
    of_3 = ObjectFifo(in_ty, name="head_3_in")
    of_4 = ObjectFifo(out_ty, name="concat_0_to_out")
    of_5 = ObjectFifo(out_ty, name="concat_1_to_out")
    of_6 = ObjectFifo(out_ty, name="out")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel acquires input tensors X and Y, and output tensor Z, performs the
    # SAXPY operation on X and Y, and writes the result in Z.

    concat_head_0_1_kernel = ExternalFunction(
        "concat1_0",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_concat.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    concat_head_2_3_kernel = ExternalFunction(
        "concat1_1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_concat.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    out_kernel = ExternalFunction(
        "out1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_out.cc"),
        arg_types=[out_ty, out_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body_2_in(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    workers = []
    workers.append(Worker(core_body_2_in, fn_args=[of_0.cons(), of_1.cons(), of_4.prod(), concat_head_0_1_kernel]))
    workers.append(Worker(core_body_2_in, fn_args=[of_2.cons(), of_3.cons(), of_5.prod(), concat_head_2_3_kernel]))
    workers.append(Worker(core_body_2_in, fn_args=[of_4.cons(), of_5.cons(), of_6.prod(), out_kernel]))
                   

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, in_ty, in_ty, in_ty, out_ty) as (in_0, in_1, in_2, in_3, out_0):
        rt.start(*workers)
        rt.fill(of_0.prod(), in_0)
        rt.fill(of_1.prod(), in_1)
        rt.fill(of_2.prod(), in_2)
        rt.fill(of_3.prod(), in_3)
        rt.drain(of_6.cons(), out_0, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())

def main():
    element_type = np.int8
    
    inp0 = np.loadtxt("./data/a1_head_0_real.txt", dtype=np.int8)
    inp1 = np.loadtxt("./data/a1_head_1_real.txt", dtype=np.int8)
    inp2 = np.loadtxt("./data/a1_head_2_real.txt", dtype=np.int8)
    inp3 = np.loadtxt("./data/a1_head_3_real.txt", dtype=np.int8)
    ref = np.loadtxt("./data/a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 40
    INPUT_COLS = 64
    OUTPUT_SIZE = 40 * 64

    if inp0.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp0.size} != {INPUT_ROWS*INPUT_COLS}")
    if inp1.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp1.size} != {INPUT_ROWS*INPUT_COLS}")
    if inp2.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp2.size} != {INPUT_ROWS*INPUT_COLS}")
    if inp3.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp3.size} != {INPUT_ROWS*INPUT_COLS}")

    # maybe don't need to tile here so may be able to delete
    # inp0_mat = inp0.reshape(INPUT_ROWS, INPUT_COLS)
    # inp0_tiled = tile_matrix(inp0_mat, 4, 8)  # flattened tiled input
    # inp1_mat = inp1.reshape(INPUT_ROWS, INPUT_COLS)
    # inp1_tiled = tile_matrix(inp1_mat, 4, 8)  # flattened tiled input
    # inp2_mat = inp2.reshape(INPUT_ROWS, INPUT_COLS)
    # inp2_tiled = tile_matrix(inp2_mat, 4, 8)  # flattened tiled input
    # inp3_mat = inp3.reshape(INPUT_ROWS, INPUT_COLS)
    # inp3_tiled = tile_matrix(inp3_mat, 4, 8)  # flattened tiled input
    

    # Convert/set Iron tensors for kernel input and output
    inp0_tensor = iron.tensor(inp0, dtype=np.int8, device="npu")
    inp1_tensor = iron.tensor(inp1, dtype=np.int8, device="npu")
    inp2_tensor = iron.tensor(inp2, dtype=np.int8, device="npu")
    inp3_tensor = iron.tensor(inp3, dtype=np.int8, device="npu")
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    mha_tail(inp0_tensor, inp1_tensor, inp2_tensor, inp3_tensor, output)

    np.savetxt("./data/a1_real.txt",
               np.array(output, dtype=np.int8),
               fmt="%d")

    out_np = np.array(output, dtype=np.int8)

    # Compare with golden output
    errors = 0
    for i, (a, r) in enumerate(zip(out_np, ref)):
        if a != r:
            print(f"Error at {i}: {a} != {r}")
            errors += 1

    if errors == 0:
        print("\nlayer 1 PASS!\n")
        sys.exit(0)
    else:
        print(f"\nError count: {errors}")
        print("layer 1 failed.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
