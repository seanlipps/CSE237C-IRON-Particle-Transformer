'''
The test a 4 layer model that is [input -> (dense) -> 2x(dense) -> (resadd) -> output]

In more detail:
Input is fed into dense_layer_0
Output of dense_layer_0 is fed into both dense_layer_1 and dense_layer_2
Output of both dense_layer_1 and dense_layer_2 are fed into resadd_layer for final output

The purpose of this is to demonstrate how ofifo can be used to connect multiple kernels.
More importantly was testing how ofifo is used to connect one kernel to multiple
'''

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
@iron.jit(is_placed=False, use_cache=False)
def dense_3_layers(input0, output0, output1):
    N = input0.shape[0]  # Tensor size
    N_out = output0.shape[0]
    element_type = output0.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_0 = ObjectFifo(in_ty, name="dense0_in")
    of_1 = ObjectFifo(out_ty, name="dense0_to_dense1-2")
    of_2 = ObjectFifo(out_ty, name="dense1_to_resadd")
    of_3 = ObjectFifo(out_ty, name="dense2_to_resadd")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    dense_ly_kernel_0 = ExternalFunction(
        "f0",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_3_layer_0.cc"),
        arg_types=[in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    dense_ly_kernel_1 = ExternalFunction(
        "f1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_3_layer_1.cc"),
        arg_types=[out_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    dense_ly_kernel_2 = ExternalFunction(
        "f2",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_3_layer_2.cc"),
        arg_types=[out_ty, out_ty],
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

    workers = []
    workers.append(Worker(core_body_1_in, fn_args=[of_0.cons(), of_1.prod(), dense_ly_kernel_0]))
    workers.append(Worker(core_body_1_in, fn_args=[of_1.cons(), of_2.prod(), dense_ly_kernel_1]))
    workers.append(Worker(core_body_1_in, fn_args=[of_1.cons(), of_3.prod(), dense_ly_kernel_2]))

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, out_ty, out_ty) as (a_x, b_y, c_z):
        rt.start(*workers)
        rt.fill(of_0.prod(), a_x)
        rt.drain(of_2.cons(), b_y, wait=True)
        rt.drain(of_3.cons(), c_z, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())

@iron.jit(is_placed=False, use_cache=False)
def resadd_1_layer(input0, input1, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_2 = ObjectFifo(in_ty, name="dense1_to_resadd")
    of_3 = ObjectFifo(in_ty, name="dense2_to_resadd")
    of_4 = ObjectFifo(out_ty, name="resadd_out")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    resadd_ly_kernel = ExternalFunction(
        "f3",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/test_3_layer_3.cc"),
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
    workers.append(Worker(core_body_2_in, fn_args=[of_2.cons(), of_3.cons(), of_4.prod(), resadd_ly_kernel]))

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, in_ty, out_ty) as (a_x, b_y, c_z):
        rt.start(*workers)
        rt.fill(of_2.prod(), a_x)
        rt.fill(of_3.prod(), b_y)
        rt.drain(of_4.cons(), c_z, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())

def main():
    element_type = np.int8
    
    inp = np.loadtxt("./iron_kernels/test_data/test_3_input.txt", dtype=np.int8)
    ref = np.loadtxt("./iron_kernels/test_data/test_3_out_ref.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 160
    INPUT_COLS = 8
    OUTPUT_SIZE = 160 * 64

    if inp.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp.size} != {INPUT_ROWS*INPUT_COLS}")

    inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input and output
    inp_tensor = iron.tensor(inp_tiled, dtype=np.int8, device="npu")
    output0 = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")
    output1 = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    dense_3_layers(inp_tensor, output0, output1)
    resadd_1_layer(output0, output1 , output)

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
