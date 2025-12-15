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
def layer_2_to_5(input0, input1, output):
    N = input0.shape[0]  # Tensor size
    N_out = output.shape[0]
    element_type = output.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_0 = ObjectFifo(in_ty, name="dense0_to_resadd2")
    of_1 = ObjectFifo(in_ty, name="mha1_to_resadd2")
    of_2 = ObjectFifo(out_ty, name="resadd2_out")
    of_3 = ObjectFifo(out_ty, name="dense_3_to_dense_4")
    of_4 = ObjectFifo(out_ty, name="dense_4_to_resadd5")
    of_5 = ObjectFifo(out_ty, name="resadd5_out")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel acquires input tensors X and Y, and output tensor Z, performs the
    # SAXPY operation on X and Y, and writes the result in Z.

    resadd_2_kernel = ExternalFunction(
        "f2",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_2.cc"),
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    dense_3_kernel = ExternalFunction(
        "f3",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_3.cc"),
        arg_types=[out_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    dense_4_kernel = ExternalFunction(
        "f4",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_4.cc"),
        arg_types=[out_ty, out_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )
    resadd_5_kernel = ExternalFunction(
        "f5",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_5.cc"),
        arg_types=[out_ty, out_ty, out_ty],
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
    workers.append(Worker(core_body_2_in, fn_args=[of_0.cons(), of_1.cons(), of_2.prod(), resadd_2_kernel]))
    workers.append(Worker(core_body_1_in, fn_args=[of_2.cons(), of_3.prod(), dense_3_kernel]))
    workers.append(Worker(core_body_1_in, fn_args=[of_3.cons(), of_4.prod(), dense_4_kernel]))
    workers.append(Worker(core_body_2_in, fn_args=[of_2.cons(), of_4.cons(), of_5.prod(), resadd_5_kernel]))
                   

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, in_ty, out_ty) as (in_0, in_1, out_0):
        rt.start(*workers)
        rt.fill(of_0.prod(), in_0)
        rt.fill(of_1.prod(), in_1)
        rt.drain(of_5.cons(), out_0, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())

def main():
    element_type = np.int8

    inp0 = np.loadtxt("./data/a0_real.txt", dtype=np.int8).flatten()
    inp1 = np.loadtxt("./data/a1_real.txt", dtype=np.int8).flatten() 
    ref = np.loadtxt("./data/a5_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 40
    INPUT_COLS = 64
    OUTPUT_SIZE = 40 * 64

    if inp0.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp0.size} != {INPUT_ROWS*INPUT_COLS}")
    if inp1.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp1.size} != {INPUT_ROWS*INPUT_COLS}")

    # Convert/set Iron tensors for kernel input and output
    inp0_tensor = iron.tensor(inp0, dtype=np.int8, device="npu")
    inp1_tensor = iron.tensor(inp1, dtype=np.int8, device="npu")
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    layer_2_to_5(inp0_tensor, inp1_tensor, output)

    np.savetxt("./data/a5_real.txt",
               np.array(output, dtype=np.int8),
               fmt="%d")

    out_np = np.array(output, dtype=np.int8)

    # Compare with golden output
    errors = 0
    for i, (a, r) in enumerate(zip(out_np, ref)):
        if a != r:
            # print(f"Error at {i}: {a} != {r}")
            errors += 1

    if errors == 0:
        print("\nlayers 2 through 5 PASS!\n")
        sys.exit(0)
    else:
        print(f"\nError count: {errors}")
        print("layers 2 through 5 failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
