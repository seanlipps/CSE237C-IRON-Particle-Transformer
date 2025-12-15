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
def mha_head_1(input0, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    qkv_ty = np.ndarray[(40*16,), np.dtype[element_type]] 
    score_ty = np.ndarray[(40*40,), np.dtype[element_type]] # because INPUT_ROWS = 40 in main()
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_0 = ObjectFifo(in_ty, name="qkv_in")
    of_1 = ObjectFifo(qkv_ty, name="q_to_score")
    of_2 = ObjectFifo(qkv_ty, name="k_to_score")
    of_3 = ObjectFifo(qkv_ty, name="v_to_context")
    of_4 = ObjectFifo(score_ty, name="score_to_context")
    of_5 = ObjectFifo(out_ty, name="context_out")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel acquires input tensors X and Y, and output tensor Z, performs the
    # SAXPY operation on X and Y, and writes the result in Z.

    head_1_q_kernel = ExternalFunction(
        "q1_head1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_q_head1.cc"),
        arg_types=[in_ty, qkv_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    head_1_k_kernel = ExternalFunction(
        "k1_head1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_k_head1.cc"),
        arg_types=[in_ty, qkv_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    head_1_v_kernel = ExternalFunction(
        "v1_head1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_v_head1.cc"),
        arg_types=[in_ty, qkv_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    head_1_scores_kernel = ExternalFunction(
        "scores1_head1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_scores_head1.cc"),
        arg_types=[qkv_ty, qkv_ty, score_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    head_1_context_kernel = ExternalFunction(
        "context1_head1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_context_head1.cc"),
        arg_types=[score_ty, qkv_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body_1_in(of_x, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_z)
        of_x.release(1)
        of_z.release(1)

    def core_body_2_in(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    workers = []
    workers.append(Worker(core_body_1_in, fn_args=[of_0.cons(), of_1.prod(), head_1_q_kernel]))
    workers.append(Worker(core_body_1_in, fn_args=[of_0.cons(), of_2.prod(), head_1_k_kernel]))
    workers.append(Worker(core_body_1_in, fn_args=[of_0.cons(), of_3.prod(), head_1_v_kernel]))
    workers.append(Worker(core_body_2_in, fn_args=[of_1.cons(), of_2.cons(), of_4.prod(), head_1_scores_kernel]))
    workers.append(Worker(core_body_2_in, fn_args=[of_4.cons(), of_3.cons(), of_5.prod(), head_1_context_kernel]))
                   

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_x, c_z):
        rt.start(*workers)
        rt.fill(of_0.prod(), a_x)
        rt.drain(of_5.cons(), c_z, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())

def main():
    element_type = np.int8
    
    inp = np.loadtxt("./data/a0_real.txt", dtype=np.int8).flatten()
    ref = np.loadtxt("./data/a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 40
    INPUT_COLS = 64
    OUTPUT_SIZE = 40 * 16

    if inp.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp.size} != {INPUT_ROWS*INPUT_COLS}")

    # maybe don't need to tile here so may be able to delete
    # inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    # inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input and output
    inp_tensor = iron.tensor(inp, dtype=np.int8, device="npu")
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    mha_head_1(inp_tensor, output)

    np.savetxt("./data/a1_head_1_real.txt",
               np.array(output, dtype=np.int8),
               fmt="%d")

    # Compare golden output would be here if it existed for a single head


if __name__ == "__main__":
    main()
