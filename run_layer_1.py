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

# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - is_placed (bool): Whether the kernel is using explicit or deferred placement API. Defaults to True.
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.

@iron.jit(is_placed=False)
def dense_q(input0, output):
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
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_q_head0.cc"),
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

@iron.jit(is_placed=False)
def dense_k(input0, output):
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
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_4_layer_1_k_head0.cc"),
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

@iron.jit(is_placed=False)
def dense_v(input0, output):
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
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_4_layer_1_v_head0.cc"),
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
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, depth=1, name="x")
    of_y = ObjectFifo(in_ty, depth=1, name="y")
    of_z = ObjectFifo(out_ty, depth=1, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    score_ly_kernel = ExternalFunction(
        "score_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_4_layer_1_scores_head0.cc"),
        arg_types=[in_tx, in_ty, out_ty],
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
    with rt.sequence(in_tx, in_ty, out_ty) as (a_x, a_y, c_z):
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
    N = input0.shape[0] # Tensor size
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    context_ly_kernel = ExternalFunction(
        "context_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_4_layer_1_context_head0.cc"),
        arg_types=[in_tx, in_ty, out_ty],
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
    with rt.sequence(in_tx, in_ty, out_ty) as (a_x, a_y, c_z):
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
def output_ly(input0, output):
    N = input0.shape[0] # Tensor size
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

    output_ly_kernel = ExternalFunction(
        "output_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_4_layer_1_out.cc"),
        arg_types=[in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    def core_body(of_x, of_z, output_ly_kernel):
        elem_x = of_x.acquire(1)
        elem_z = of_z.acquire(1)
        output_ly_kernel(elem_x, elem_z)
        of_x.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_z.prod(), output_ly_kernel]
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
    element_type = np.int8
    
    inp = np.loadtxt("./iron_kernels/data/a0_real.txt", dtype=np.int8).flatten()
    ref = np.loadtxt("./iron_kernels/data/a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 40
    ff_dim = 64
    OUTPUT_SIZE = 40 * 64

    # Convert/set Iron tensors for kernel input
    inp_tensor = iron.tensor(inp, dtype=np.int8, device="npu")

    q_output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu") #40x64 -> 40x64
    k_output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu") #40x64 -> 40x64
    v_output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu") #40x64 -> 40x64

    score_output = iron.zeros(INPUT_ROWS*INPUT_ROWS, dtype=element_type, device="npu") #40x64x40 -> 40x40
    context_output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu") #40x40x64 -> 40x64
    
    output_output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu") #40x64 -> 40x64

    # Insantiate AIE Kernel        
    # for i in range(4):
    dense_q(inp_tensor, q_output)
    dense_k(inp_tensor, k_output)
    dense_v(inp_tensor, v_output)
    
    score_ly(q_output, k_output, score_output)
    print("scores_output: ", score_output);
    context_ly(score_output, v_output, context_output)
    print("context_output: ", context_output);
    output_ly(context_output, output_output)
    print("output_output: ", output_output);

    np.savetxt("q_output.txt",
               np.array(q_output, dtype=np.int8),
               fmt="%d")
    np.savetxt("k_output.txt",
               np.array(k_output, dtype=np.int8),
               fmt="%d")
    np.savetxt("scores_output.txt",
               np.array(score_output, dtype=np.int8),
               fmt="%d")
    np.savetxt("context_output.txt",
               np.array(context_output, dtype=np.int8),
               fmt="%d")
    np.savetxt("output_output.txt",
               np.array(output_output, dtype=np.int8),
               fmt="%d")
    
    out_np = np.array(output_output, dtype=np.int8)

    errors = 0
    for i, (a, r) in enumerate(zip(out_np, ref)):
        if a != r:
            # print(f"Error at {i}: {a} != {r}")
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
