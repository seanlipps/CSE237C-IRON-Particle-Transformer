
import numpy as np
import os
import sys
import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.utils.config import cxx_header_path

SOURCE_FILE = r"/notebooks/CSE237C-IRON-Particle-Transformer/iron_kernels/test_4_layer_1_concat.cc"
INCLUDE_DIR = r"/notebooks/CSE237C-IRON-Particle-Transformer/iron_kernels"

@iron.jit(is_placed=False)
def concat1_1_impl(input0, input1, output):
    N = input0.shape[0]
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    kernel = ExternalFunction(
        "concat1_1",
        source_file=SOURCE_FILE,
        arg_types=[in_tx, in_ty, out_ty],
        include_dirs=[cxx_header_path(), INCLUDE_DIR],
    )

    def core_body(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), kernel])

    rt = Runtime()
    with rt.sequence(in_tx, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())

if __name__ == "__main__":
    # Load Inputs
    input0_data = iron.tensor(np.loadtxt(r"/notebooks/CSE237C-IRON-Particle-Transformer/temp_concat1_1_in0.txt", dtype=np.int8).flatten(), dtype=np.int8, device="npu")
    input1_data = iron.tensor(np.loadtxt(r"/notebooks/CSE237C-IRON-Particle-Transformer/temp_concat1_1_in1.txt", dtype=np.int8).flatten(), dtype=np.int8, device="npu")


    output_placeholder = iron.zeros((1280,), dtype=np.int8, device="npu")
    
    concat1_1_impl(input0_data, input1_data, output_placeholder)

    np.savetxt(r"/notebooks/CSE237C-IRON-Particle-Transformer/temp_concat1_1_out.txt", np.array(output_placeholder, dtype=np.int8), fmt="%d")
