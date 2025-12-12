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


# ------------------------------------------------------------------------------
# dense layer (Q/K/V projection)
# ------------------------------------------------------------------------------

def make_dense_ly(qkv: str, mha_num: int, head_num: int):
    """
    Create a jit-compiled dense layer function for a specific (qkv, mha_num, head_num).
    Returns a function dense_ly(input0, output).
    """
    kernel_name = f"{qkv}{mha_num}_head{head_num}"
    kernel_relpath = f"iron_kernels/layer_{mha_num}_{qkv}_head{head_num}.cc"
    kernel_path = os.path.join(os.path.dirname(__file__), kernel_relpath)

    @iron.jit(is_placed=False)
    def dense_ly(input0, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        # Tensor types
        in_ty = np.ndarray[(N,), np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        # FIFOs
        of_x = ObjectFifo(in_ty, name="x")
        of_z = ObjectFifo(out_ty, name="z")

        # External AIE kernel
        dense_ly_kernel = ExternalFunction(
            kernel_name,
            source_file=kernel_path,
            arg_types=[in_ty, out_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels"),
            ],
        )

        def core_body(of_x, of_z, dense_ly_kernel):
            elem_x = of_x.acquire(1)
            elem_z = of_z.acquire(1)
            dense_ly_kernel(elem_x, elem_z)
            of_x.release(1)
            of_z.release(1)

        worker = Worker(core_body, fn_args=[of_x.cons(), of_z.prod(), dense_ly_kernel])

        # Runtime and data movement
        rt = Runtime()
        with rt.sequence(in_ty, out_ty) as (a_x, c_z):
            rt.start(worker)
            rt.fill(of_x.prod(), a_x)
            rt.drain(of_z.cons(), c_z, wait=True)

        # Program + placement
        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    return dense_ly


# ------------------------------------------------------------------------------
# score layer (Q @ K^T)
# ------------------------------------------------------------------------------

def make_score_ly(mha_num: int, head_num: int):
    """
    Create a jit-compiled score layer (QK^T) for a specific (mha_num, head_num).
    Returns score_ly(input0, input1, output).
    """
    kernel_name = f"scores{mha_num}_head{head_num}"
    kernel_relpath = f"iron_kernels/layer_{mha_num}_scores_head{head_num}.cc"
    kernel_path = os.path.join(os.path.dirname(__file__), kernel_relpath)

    @iron.jit(is_placed=False)
    def score_ly(input0, input1, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        in_ty = np.ndarray[(N,), np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_x = ObjectFifo(in_ty, depth=1, name="x")
        of_y = ObjectFifo(in_ty, depth=1, name="y")
        of_z = ObjectFifo(out_ty, depth=1, name="z")

        score_ly_kernel = ExternalFunction(
            kernel_name,
            source_file=kernel_path,
            arg_types=[in_ty, in_ty, out_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels"),
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

        rt = Runtime()
        with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
            rt.start(worker)
            rt.fill(of_x.prod(), a_x)
            rt.fill(of_y.prod(), a_y)
            rt.drain(of_z.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    return score_ly


# ------------------------------------------------------------------------------
# context layer (scores @ V)
# ------------------------------------------------------------------------------

def make_context_ly(mha_num: int, head_num: int):
    """
    Create a jit-compiled context layer (scores @ V) for (mha_num, head_num).
    Returns context_ly(input0, input1, output).
    """
    kernel_name = f"context{mha_num}_head{head_num}"
    kernel_relpath = f"iron_kernels/layer_{mha_num}_context_head{head_num}.cc"
    kernel_path = os.path.join(os.path.dirname(__file__), kernel_relpath)

    @iron.jit(is_placed=False)
    def context_ly(input0, input1, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        in_ty = np.ndarray[(N,), np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_x = ObjectFifo(in_ty, depth=1, name="x")
        of_y = ObjectFifo(in_ty, depth=1, name="y")
        of_z = ObjectFifo(out_ty, depth=1, name="z")

        context_ly_kernel = ExternalFunction(
            kernel_name,
            source_file=kernel_path,
            arg_types=[in_ty, in_ty, out_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels"),
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

        rt = Runtime()
        with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
            rt.start(worker)
            rt.fill(of_x.prod(), a_x)
            rt.fill(of_y.prod(), a_y)
            rt.drain(of_z.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    return context_ly


# ------------------------------------------------------------------------------
# concat layer (concatenate heads)
# ------------------------------------------------------------------------------

def make_concat_ly(concat_num: int, mha_num: int):
    """
    Create a jit-compiled concat layer for (mha_num, concat_num).
    Returns concat_ly(input0, input1, output).
    """
    kernel_name = f"concat{mha_num}_{concat_num}"
    kernel_relpath = f"iron_kernels/layer_{mha_num}_concat.cc"
    kernel_path = os.path.join(os.path.dirname(__file__), kernel_relpath)

    @iron.jit(is_placed=False)
    def concat_ly(input0, input1, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        in_ty = np.ndarray[(N,), np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_x = ObjectFifo(in_ty, depth=1, name="x")
        of_y = ObjectFifo(in_ty, depth=1, name="y")
        of_z = ObjectFifo(out_ty, depth=1, name="z")

        concat_ly_kernel = ExternalFunction(
            kernel_name,
            source_file=kernel_path,
            arg_types=[in_ty, in_ty, out_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels"),
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

        rt = Runtime()
        with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
            rt.start(worker)
            rt.fill(of_x.prod(), a_x)
            rt.fill(of_y.prod(), a_y)
            rt.drain(of_z.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    return concat_ly


# ------------------------------------------------------------------------------
# output projection layer (context @ Wo)
# ------------------------------------------------------------------------------

def make_output_ly(mha_num: int):
    """
    Create a jit-compiled output layer (context @ Wo) for mha_num.
    Returns output_ly(input0, input1, output).
    """

    kernel_name = f"out{mha_num}"
    kernel_relpath = f"iron_kernels/layer_{mha_num}_out.cc"
    kernel_path = os.path.join(os.path.dirname(__file__), kernel_relpath)

    @iron.jit(is_placed=False)
    def output_ly(input0, input1, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        in_ty = np.ndarray[(N,), np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_x = ObjectFifo(in_ty, depth=1, name="x")
        of_y = ObjectFifo(in_ty, depth=1, name="y")
        of_z = ObjectFifo(out_ty, depth=1, name="z")

        output_ly_kernel = ExternalFunction(
            kernel_name,   # or "output_kernel" if that’s what C++ expects
            source_file=kernel_path,
            arg_types=[in_ty, in_ty, out_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels"),
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

        rt = Runtime()
        with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
            rt.start(worker)
            rt.fill(of_x.prod(), a_x)
            rt.fill(of_y.prod(), a_y)
            rt.drain(of_z.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    return output_ly

def main():
    element_type = np.int8

    inp = np.loadtxt("./data/a0_golden.txt", dtype=np.int8)
    ref = np.loadtxt("./data/a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 160
    INPUT_COLS = 64

    if inp.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp.size} != {INPUT_ROWS * INPUT_COLS}")

    inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    #inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input
    inp_tensor = iron.tensor(inp_mat, dtype=np.int8, device="npu")

    q_output = [iron.zeros(160 * 16, dtype=element_type, device="npu") for _ in range(4)]
    k_output = [iron.zeros(160 * 16, dtype=element_type, device="npu") for _ in range(4)]
    v_output = [iron.zeros(160 * 16, dtype=element_type, device="npu") for _ in range(4)]

    score_output = [iron.zeros(160 * 160, dtype=element_type, device="npu") for _ in range(4)]
    context_output = [iron.zeros(160 * 16, dtype=element_type, device="npu") for _ in range(4)]

    concat_output = [iron.zeros(160 * 32, dtype=element_type, device="npu") for _ in range(2)]
    output_output = iron.zeros(160 * 64, dtype=element_type, device="npu")

    #------------------------------function argument------------------------- #
    # make_dense_ly(qkv, mha_num, head_num) 
    # make_score_ly(mha_num, head_num) 
    # make_context_ly(mha_num, head_num) 
    # make_concat_ly(concat_num, mha_num) 
    # make_output_ly(mha_num) 
    
    # qkv (string) = tells whether matmul is done on q, k, or v 
    # mha_num (int) = mha layer number (1-indexed) 
    # head_num (int) = head number (0-indexed) 
    # concat_num (int) = concat number (0-indexed)

    
    # Instantiate AIE kernels per head / layer
    dense_q = [make_dense_ly("q", 1, i) for i in range(4)]
    dense_k = [make_dense_ly("k", 1, i) for i in range(4)]
    dense_v = [make_dense_ly("v", 1, i) for i in range(4)]

    score_heads = [make_score_ly(1, i) for i in range(4)]
    context_heads = [make_context_ly(1, i) for i in range(4)]

    concat_fns = [make_concat_ly(0, 1), make_concat_ly(1, 1)]
    output_fn = make_output_ly(1)

    # ------------------------ Run kernels ------------------------

    for i in range(4):
        # 160*64 @ 64*16 = 160*16 (tiled)
        dense_q[i](inp_tensor, q_output[i])
        dense_k[i](inp_tensor, k_output[i])
        dense_v[i](inp_tensor, v_output[i])

        # 160*16 @ 160*16^T = 160*160
        score_heads[i](q_output[i], k_output[i], score_output[i])

        # 160*160 @ 160*16 = 160*16
        context_heads[i](score_output[i], v_output[i], context_output[i])

    # concatenate heads: (0,1) and (2,3) -----> 160*16 concat 160*16 = 160*32
    concat_fns[0](context_output[0], context_output[1], concat_output[0])  # 160*32
    concat_fns[1](context_output[2], context_output[3], concat_output[1])  # 160*32

    # output layer: 160*64
    output_fn(concat_output[0], concat_output[1], output_output)

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
