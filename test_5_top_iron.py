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
# Dense 
# compute dense for mha layer input
# Compute Tile Utilization: 1/16
def make_dense_ly(layer_num: int):
    def dense_ly(input0, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype
        
        in_ty  = np.ndarray[(N,),     np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_x = ObjectFifo(in_ty,  name="x")
        of_z = ObjectFifo(out_ty, name="z")

        dense_ly_kernel = ExternalFunction(
            f"f{layer_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}.cc"),
            arg_types=[in_ty, out_ty],
            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )

        def core_body(of_x, of_z, dense_ly_kernel):
            elem_x = of_x.acquire(1)
            elem_z = of_z.acquire(1)
            dense_ly_kernel(elem_x, elem_z)
            of_x.release(1)
            of_z.release(1)

        worker = Worker(core_body, fn_args=[of_x.cons(), of_z.prod(), dense_ly_kernel])

        rt = Runtime()
        with rt.sequence(in_ty, out_ty) as (a_x, c_z):
            rt.start(worker)
            rt.fill(of_x.prod(), a_x)
            rt.drain(of_z.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    dense_ly.__name__ = f"dense_ly_{layer_num}"
    return iron.jit(is_placed=False)(dense_ly)


##############################################################
# MHA part 1 (single head)
# takes in dense output
# compute q, k, v 
# compute score
# compute context
# Compute Tile Utilization: 5/16
def make_mha_p1(layer_num: int, head_num: int):    
    def mha_p1(input0, output):
        N = input0.shape[0]   # 160 * 64
        N_out = output.shape[0] # 160 * 16
        element_type = output.dtype
        
        
        # Tensor types
        in_ty = np.ndarray[(N,), np.dtype[element_type]]
        qkv_ty = np.ndarray[(160*16,), np.dtype[element_type]]
        score_ty = np.ndarray[(160*160,), np.dtype[element_type]]
        context_ty = np.ndarray[(160*16,), np.dtype[element_type]]
    
        # FIFOs
        of_in      = ObjectFifo(in_ty, name=f"in_h{head_num}", depth=1)
        of_q_out   = ObjectFifo(qkv_ty, name=f"q_out_h{head_num}", depth=1)
        of_k_out   = ObjectFifo(qkv_ty, name=f"k_out_h{head_num}", depth=1)
        of_v_out   = ObjectFifo(qkv_ty, name=f"v_out_h{head_num}", depth=1)
        of_score   = ObjectFifo(score_ty, name=f"score_out_h{head_num}", depth=1)
        of_context = ObjectFifo(context_ty, name=f"context_out_h{head_num}", depth=1)
    
        # Kernels
        q_kernel = ExternalFunction(
            f"q{layer_num}_head{head_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_q_head{head_num}.cc"),
            arg_types=[in_ty, qkv_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels")
            ],
        )
        
        k_kernel = ExternalFunction(
            f"k{layer_num}_head{head_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_k_head{head_num}.cc"),
            arg_types=[in_ty, qkv_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels")
            ],
        )

        v_kernel = ExternalFunction(
            f"v{layer_num}_head{head_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_v_head{head_num}.cc"),
            arg_types=[in_ty, qkv_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels")
            ],
        )
        
        score_kernel = ExternalFunction(
            f"scores{layer_num}_head{head_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_scores_head{head_num}.cc"),
            arg_types=[qkv_ty, qkv_ty, score_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels")
            ],
        )

        context_kernel = ExternalFunction(
            f"context{layer_num}_head{head_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_context_head{head_num}.cc"),
            arg_types=[score_ty, qkv_ty, context_ty],
            include_dirs=[
                cxx_header_path(),
                os.path.join(os.path.dirname(__file__), "iron_kernels")
            ],
        )
        
        # task for one input, one output
        def core_body_dense (of_in, of_out, kernel):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out)
            of_in.release(1)
            of_out.release(1)
    
        # task for two inputs, one output
        def core_body_in2 (of_in0, of_in1, of_out, kernel):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in0, elem_in1, elem_out)
            of_in0.release(1)
            of_in1.release(1)
            of_out.release(1)
    
        workers = []
        workers.append(Worker(core_body_dense, fn_args=[of_in.cons(), of_q_out.prod(), q_kernel]))
        workers.append(Worker(core_body_dense, fn_args=[of_in.cons(), of_k_out.prod(), k_kernel]))
        workers.append(Worker(core_body_dense, fn_args=[of_in.cons(), of_v_out.prod(), v_kernel]))
        workers.append(Worker(core_body_in2, fn_args=[of_q_out.cons(), of_k_out.cons(), of_score.prod(), score_kernel]))
        workers.append(Worker(core_body_in2, fn_args=[of_score.cons(), of_v_out.cons(), of_context.prod(), context_kernel]))
        
        # Runtime and data movement
        rt = Runtime()
        with rt.sequence(in_ty, context_ty) as (a_x, c_z):
            rt.start(*workers)
            rt.fill(of_in.prod(), a_x)
            rt.drain(of_context.cons(), c_z, wait=True)
    
        # Program + placement
        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    mha_p1.__name__ = f"mha_p1_{layer_num}_h{head_num}"
        
    return iron.jit(is_placed=False)(mha_p1)


##############################################################
# MHA part 2
# takes in context outputs from 4 heads
# concat head0,1 and head2,3
# output
# Compute Tile Utilization: 3/16
def make_mha_p2(layer_num: int):
    def mha_p2(input0, input1, input2, input3, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        in_ty     = np.ndarray[(N,), np.dtype[element_type]]
        concat_ty = np.ndarray[(160*32,), np.dtype[element_type]]
        out_ty    = np.ndarray[(N_out,), np.dtype[element_type]]

        of_in     = [ObjectFifo(in_ty, name=f"in_{i}", depth=1) for i in range(4)]
        of_concat = [ObjectFifo(concat_ty, name=f"concat_{i}", depth=1) for i in range(2)]
        of_out    = ObjectFifo(out_ty, name="out", depth=1)
        
        concat_kernels = [
            ExternalFunction(
                f"concat{layer_num}_{i}",
                source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_concat.cc"),
                arg_types=[in_ty, in_ty, concat_ty],
                include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
            ) for i in range(2)
        ]

        out_kernel = ExternalFunction(
            f"out{layer_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}_out.cc"),
            arg_types=[concat_ty, concat_ty, out_ty],
            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )

        def core_body_in2(of_in0, of_in1, of_out, kernel):
            e0 = of_in0.acquire(1)
            e1 = of_in1.acquire(1)
            eo = of_out.acquire(1)
            kernel(e0, e1, eo)
            of_in0.release(1)
            of_in1.release(1)
            of_out.release(1)

        workers = [
            Worker(core_body_in2, fn_args=[of_in[0].cons(), of_in[1].cons(), of_concat[0].prod(), concat_kernels[0]]),
            Worker(core_body_in2, fn_args=[of_in[2].cons(), of_in[3].cons(), of_concat[1].prod(), concat_kernels[1]]),
            Worker(core_body_in2, fn_args=[of_concat[0].cons(), of_concat[1].cons(), of_out.prod(), out_kernel]),
        ]

        rt = Runtime()
        with rt.sequence(in_ty, in_ty, in_ty, in_ty, out_ty) as (a0_x, a1_x, a2_x, a3_x, c_z):
            rt.start(*workers)
            rt.fill(of_in[0].prod(), a0_x)
            rt.fill(of_in[1].prod(), a1_x)
            rt.fill(of_in[2].prod(), a2_x)
            rt.fill(of_in[3].prod(), a3_x)
            rt.drain(of_out.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    mha_p2.__name__ = f"mha_p2_{layer_num}"
    return iron.jit(is_placed=False)(mha_p2)

    
##############################################################
# Resadd + dense + dense + resadd
# takes in dense output and mha output
# Compute Tile Utilization: 4/16
def make_resadd_ly(layer_num: int):
    def resadd_ly(input0, input1, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype

        in_ty  = np.ndarray[(N,),     np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_in0 = ObjectFifo(in_ty, name="in0")
        of_in1 = ObjectFifo(in_ty, name="in1")
        of_res = ObjectFifo(in_ty, name="res")
        of_ffa = ObjectFifo(in_ty, name="ffa")
        of_ffb = ObjectFifo(in_ty, name="ffb")
        of_out = ObjectFifo(out_ty, name="out")

        resadd_ly_kernel0 = ExternalFunction(
            f"f{layer_num}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}.cc"),
            arg_types=[in_ty, in_ty, out_ty],
            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )
        dense_ly_kernel0 = ExternalFunction(
            f"f{layer_num+1}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num+1}.cc"),
            arg_types=[in_ty, out_ty],
            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )
        dense_ly_kernel1 = ExternalFunction(
            f"f{layer_num+2}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num+2}.cc"),
            arg_types=[in_ty, out_ty],
            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )
        resadd_ly_kernel1 = ExternalFunction(
            f"f{layer_num+3}",
            source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num+3}.cc"),
            arg_types=[in_ty, in_ty, out_ty],
            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )

        def core_body_dense(of_in, of_out, kernel):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            kernel(ei, eo)
            of_in.release(1)
            of_out.release(1)

        def core_body_in2(of_x, of_y, of_z, kernel):
            ex = of_x.acquire(1)
            ey = of_y.acquire(1)
            ez = of_z.acquire(1)
            kernel(ex, ey, ez)
            of_x.release(1)
            of_y.release(1)
            of_z.release(1)

        workers = [
            Worker(core_body_in2,   fn_args=[of_in0.cons(), of_in1.cons(), of_res.prod(), resadd_ly_kernel0]),
            Worker(core_body_dense, fn_args=[of_res.cons(), of_ffa.prod(), dense_ly_kernel0]),
            Worker(core_body_dense, fn_args=[of_ffa.cons(), of_ffb.prod(), dense_ly_kernel1]),
            Worker(core_body_in2,   fn_args=[of_ffb.cons(), of_res.cons(), of_out.prod(), resadd_ly_kernel1]),
        ]

        rt = Runtime()
        with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
            rt.start(*workers)
            rt.fill(of_in0.prod(), a_x)
            rt.fill(of_in1.prod(), a_y)
            rt.drain(of_out.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    resadd_ly.__name__ = f"resadd_ly_{layer_num}"
    return iron.jit(is_placed=False)(resadd_ly)


########################################
# Dense + Dense
# compute dense in series
# Compute Tile Utilization: 2/16
def make_two_dense_ly(layer_num: int):
    def two_dense_ly(input0, output):
        N = input0.shape[0]
        N_out = output.shape[0]
        element_type = output.dtype
        
        in_ty  = np.ndarray[(N,),     np.dtype[element_type]]
        out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

        of_x = ObjectFifo(in_ty, name="x")
        of_y = ObjectFifo(in_ty, name="y")
        of_z = ObjectFifo(out_ty, name="z")

        dense_ly_kernel0 = ExternalFunction(
                f"f{layer_num}",
                source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num}.cc"),
                arg_types=[in_ty, in_ty],
                include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
            )

        dense_ly_kernel1 = ExternalFunction(
                f"f{layer_num+1}",
                source_file=os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_{layer_num+1}.cc"),
                arg_types=[in_ty, out_ty],
                include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")],
        )

        def core_body(of_x, of_z, kernel):
            ex = of_x.acquire(1)
            ez = of_z.acquire(1)
            kernel(ex, ez)
            of_x.release(1)
            of_z.release(1)

        workers = []
        workers.append(Worker(core_body, fn_args=[of_x.cons(), of_y.prod(), dense_ly_kernel0]))
        workers.append(Worker(core_body, fn_args=[of_y.cons(), of_z.prod(), dense_ly_kernel1]))

        rt = Runtime()
        with rt.sequence(in_ty, out_ty) as (a_x, c_z):
            rt.start(*workers)
            rt.fill(of_x.prod(), a_x)
            rt.drain(of_z.cons(), c_z, wait=True)

        my_program = Program(iron.get_current_device(), rt)
        return my_program.resolve_program(SequentialPlacer())

    two_dense_ly.__name__ = f"two_dense_ly_{layer_num}"
    return iron.jit(is_placed=False)(two_dense_ly)


def main():
    element_type = np.int8
    
    INPUT_ROWS = 160
    INPUT_COLS = 8

    inp = np.loadtxt("./data/input.txt", dtype=np.int8)
    ref = np.loadtxt("./data/out_ref.txt", dtype=np.int8).flatten()

    if inp.size != INPUT_ROWS * INPUT_COLS:
        raise ValueError(f"input size {inp.size} != {INPUT_ROWS*INPUT_COLS}")

    inp_mat = inp.reshape(INPUT_ROWS, INPUT_COLS)
    inp_tiled = tile_matrix(inp_mat, 4, 8)  # flattened tiled input

    # Convert/set Iron tensors for kernel input and output
    inp_tensor = iron.tensor(inp_tiled, dtype=np.int8, device="npu")
    dense_out_tensor = iron.zeros(160*64, dtype=element_type, device="npu")
    mha1_p1_out_tensor = [iron.zeros(160*16, dtype=element_type, device="npu") for i in range(4)]
    mha1_p2_out_tensor = iron.zeros(160*64, dtype=element_type, device="npu")
    resdense1_out_tensor = iron.zeros(160*64, dtype=element_type, device="npu")
    mha2_p1_out_tensor = [iron.zeros(160*16, dtype=element_type, device="npu") for i in range(4)]
    mha2_p2_out_tensor = iron.zeros(160*64, dtype=element_type, device="npu")
    resdense2_out_tensor = iron.zeros(160*64, dtype=element_type, device="npu")
    output = iron.zeros(160*8, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    dense_fn = make_dense_ly(0)                           # layer 0
    mha1_p1_fn = [make_mha_p1(1, i) for i in range(4)]    # layer 1
    mha1_p2_fn = make_mha_p2(1)                           # layer 1
    resadd1_fn = make_resadd_ly(2)                        # layer 2,3,4,5
    mha2_p1_fn = [make_mha_p1(6, i) for i in range(4)]    # layer 6
    mha2_p2_fn = make_mha_p2(6)                           # layer 6
    resadd2_fn = make_resadd_ly(7)                        # layer 7,8,9,10 
    two_dense_fn = make_two_dense_ly(11)                  # layer 11, 12
    
    # Run kernels (Data flow)
    dense_fn(inp_tensor, dense_out_tensor)                         # layer 0
    for i in range(4):     
        mha1_p1_fn[i](dense_out_tensor, mha1_p1_out_tensor[i])     # layer 1    
    mha1_p2_fn(mha1_p1_out_tensor[0], mha1_p1_out_tensor[1], mha1_p1_out_tensor[2], mha1_p1_out_tensor[3], mha1_p2_out_tensor)
    resadd1_fn(dense_out_tensor, mha1_p2_out_tensor, resdense1_out_tensor)      # layer 2,3,4,5
    for i in range(4): 
        mha2_p1_fn[i](resdense1_out_tensor, mha2_p1_out_tensor[i])  # layer 6    
    mha2_p2_fn(mha2_p1_out_tensor[0], mha2_p1_out_tensor[1], mha2_p1_out_tensor[2], mha2_p1_out_tensor[3], mha2_p2_out_tensor)
    resadd2_fn(resdense1_out_tensor, mha2_p2_out_tensor, resdense2_out_tensor)  # layer 7,8,9,10
    two_dense_fn(resdense2_out_tensor, output)                      # layer 11, 12


    import os
    os.makedirs("./data/sim", exist_ok=True)
    np.savetxt("./data/sim/a0_sim.txt", np.array(dense_out_tensor, dtype=np.int8).reshape(-1, 16), fmt="%d")
    np.savetxt("./data/sim/a1_sim.txt", np.array(mha1_p2_out_tensor, dtype=np.int8).reshape(-1, 16), fmt="%d")
    np.savetxt("./data/sim/a5_sim.txt", np.array(resdense1_out_tensor, dtype=np.int8).reshape(-1, 16), fmt="%d")
    np.savetxt("./data/sim/a6_sim.txt", np.array(mha2_p2_out_tensor, dtype=np.int8).reshape(-1, 16), fmt="%d")
    np.savetxt("./data/sim/a10_sim.txt", np.array(resdense2_out_tensor, dtype=np.int8).reshape(-1, 16), fmt="%d")
    np.savetxt("./data/out_sim.txt", np.array(output, dtype=np.int8).reshape(-1, 16), fmt="%d")


if __name__ == "__main__":
    main()
