
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
        "dense_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/dense_layer.cc"),
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



######################################################################################
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

######################################################################################
# context
@iron.jit(is_placed=False)
def context_ly(input0, input1, output):
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

    context_ly_kernel = ExternalFunction(
        "context_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/context_layer.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_y, of_z, context_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        context_ly_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), context_ly_kernel]
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


######################################################################################
# concat
@iron.jit(is_placed=False)
def concat_ly(input0, input1, output):
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

    concat_ly_kernel = ExternalFunction(
        "concat_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/concat_layer.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_y, of_z, concat_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        concat_ly_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), concat_ly_kernel]
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


######################################################################################
# output
@iron.jit(is_placed=False)
def output_ly(input0, input1, output):
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

    output_ly_kernel = ExternalFunction(
        "output_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/output_layer.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_y, of_z, output_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        output_ly_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), output_ly_kernel]
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
    
    inp = np.loadtxt("./data/input.txt", dtype=np.int8)
    ref = np.loadtxt("./data/a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 160
    INPUT_COLS = 8
    OUTPUT_SIZE = 160 * 64

    if inp.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp.size} != {INPUT_ROWS*INPUT_COLS}")

    inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input
    inp_tensor = iron.tensor(inp_tiled, dtype=np.int8, device="npu")

    q_output = [iron.zeros(160*64, dtype=element_type, device="npu") for _ in range(4)]
    k_output = [iron.zeros(160*64, dtype=element_type, device="npu") for _ in range(4)]
    v_output = [iron.zeros(160*64, dtype=element_type, device="npu") for _ in range(4)]

    score_output = [iron.zeros(160*160, dtype=element_type, device="npu") for _ in range(4)]
    context_output = [iron.zeros(160*64, dtype=element_type, device="npu") for _ in range(4)]
    
    concat_output = [iron.zeros(160*128, dtype=element_type, device="npu") for _ in range(2)]
    output_output = iron.zeros(160*64, dtype=element_type, device="npu")

    # Insantiate AIE Kernel        
    for i in range(4):
        dense_ly(inp_tensor, q_output[i]) # 160*64 @ 64*16 = 160*16
        dense_ly(inp_tensor, k_output[i]) # 160*64 @ 64*16 = 160*16
        dense_ly(inp_tensor, v_output[i]) # 160*64 @ 64*16 = 160*16
        
        score_ly(q_output[0], k_output[0], score_output[i]) # 160*16 @ 160*16^T = 160*16 @ 16*160 = 160*160
        context_ly(score_output[i], v_output[i], context_output[i]) # 160*160 @ 160*16 = 160*16

    # concatenating two times: head0 + head1 and  head2 + head3
    concat_ly(context_output[0], context_output[1], concat_output[0]) # 160*16 concat 160*16 = 160*32
    concat_ly(context_output[2], context_output[3], concat_output[1]) # 160*16 concat 160*16 = 160*32

    # output layer
    output_ly(concat_output[0], concat_output[1], output_output) # 160*64
    
    out_np = np.array(output_output, dtype=np.int8)

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
