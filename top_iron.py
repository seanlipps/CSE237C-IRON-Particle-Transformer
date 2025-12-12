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

@iron.jit(is_placed=False)
def particle_transformer(input0, output):
    N = input0.shape[0]   # 160 * 8
    N_out = output.shape[0] # 160 * 64
    element_type = output.dtype

    # Tensor types
    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    mha_in_ty = np.ndarray[(160*64,), np.dtype[element_type]]
    qkv_ty = np.ndarray[(160*16,), np.dtype[element_type]]
    score_ty = np.ndarray[(160*160,), np.dtype[element_type]]
    context_ty = np.ndarray[(160*16,), np.dtype[element_type]]
    concat_ty = np.ndarray[(160*32,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    
    # Tile layout (row, col):
    #
    #       col=0         col=1             col=2             col=3
    #      -------------------------------------------------------------
    # r=5 | context[0]|   concat[0]  |   concat[1]  |    out      |
    # r=4 | score[0]  |  context[1]  |   context[2] |  context[3] |
    # r=3 | qkv[0]    |   score[1]   |    score[2]  |   score[3]  |
    # r=2 | dense     |    qkv[1]    |    qkv[2]    |    qkv[3]   |
    # r=1 |   mem1    |      mem2    |     mem3     |    mem4     |
    # r=0 |   shim1   |     shim2    |     shim3    |   shim4     |

    # shim1 = (0,0)     (col, row)
    # mem1 = (0,1) 

    ################### tile placement ####################    
    # shim_tile  = tile(0,0)
    # dense_tile = tile(0,2)
    
    # # QKV tiles, one per head
    # qkv_tile = [
    #     tile(0, 3),  # qkv[0]
    #     tile(1, 2),  # qkv[1]
    #     tile(2, 2),  # qkv[2]
    #     tile(3, 2),  # qkv[3]
    # ]
    
    # # Score tiles
    # score_tile = [
    #     tile(0, 4),  # score[0]
    #     tile(1, 3),  # score[1]
    #     tile(2, 3),  # score[2]
    #     tile(3, 3),  # score[3]
    # ]
    
    # # Context tiles
    # context_tile = [
    #     tile(0, 5),  # context[0]
    #     tile(1, 4),  # context[1]
    #     tile(2, 4),  # context[2]
    #     tile(3, 4),  # context[3]
    # ]
    
    # # Concat tiles
    # concat_tile = [
    #     tile(1, 5),  # concat[0]
    #     tile(2, 5),  # concat[1]
    # ]
    
    # # Out tile: out at (3,5)
    # out_tile = tile(3, 5)


    # FIFOs
    of_in = ObjectFifo(in_ty, name="in", depth=1)    
    of_mha_in = ObjectFifo(mha_in_ty, name="mha_in", depth=1)    
    of_q_out = [ObjectFifo(qkv_ty, name = f"q_out_{i}", depth=1) for i in range(4)]
    of_k_out = [ObjectFifo(qkv_ty, name = f"k_out_{i}", depth=1) for i in range(4)]
    of_v_out = [ObjectFifo(qkv_ty, name = f"v_out_{i}", depth=1) for i in range(4)]
    of_score = [ObjectFifo(score_ty, name = f"score_{i}", depth=1) for i in range(4)]
    of_context = [ObjectFifo(context_ty, name = f"context_{i}", depth=1) for i in range(4)]
    of_concat = [ObjectFifo(concat_ty, name = f"concat_{i}", depth=1) for i in range(2)]
    of_out = ObjectFifo(out_ty, name = "out", depth=1)

    # of_in = object_fifo("in", shim_tile, dense_tile, 1, in_ty)    
    # of_mha_in = object_fifo("mha_in", dense_tile, qkv_tile, 1, mha_in_ty)    
    # of_q_out = [object_fifo(f"q_out_{i}", q_tile[i], score_tile[i], 1, qkv_ty) for i in range(4)]
    # of_k_out = [object_fifo(f"k_out_{i}", k_tile[i], score_tile[i], 1, qkv_ty) for i in range(4)]
    # of_v_out = [object_fifo(f"v_out_{i}", v_tile[i], context_tile[i], 1, qkv_ty) for i in range(4)]
    # of_score = [object_fifo(f"score_{i}", score_tile[i], context_tile[i], 1, score_ty) for i in range(4)]
    # of_context = [object_fifo(f"context_{i}", context_tile[i], concat_tile[i//2], 1, context_ty) for i in range(4)]
    # of_concat = [object_fifo(f"concat_{i}", concat_tile[i], out_tile, 1, concat_ty) for i in range(2)]
    # of_out = object_fifo("out", out_tile, shim_tile, 1, out_ty)

    # Kernels
    dense_ly_kernel = ExternalFunction(
        "f0",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_0.cc"),
        arg_types=[in_ty, mha_in_ty],
        include_dirs=[
            cxx_header_path(),
            os.path.join(os.path.dirname(__file__), "iron_kernels")
        ],
    )

    qkv_kernels = [ExternalFunction(f"qkv1_head{i}",
                            source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_qkv_head{i}.cc"),
                            arg_types=[mha_in_ty, qkv_ty, qkv_ty, qkv_ty], 
                            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
                            ],
                   ) for i in range (4)]

    # q_kernels = [ExternalFunction(f"q1_head{i}",
    #                         source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_q_head{i}.cc"),
    #                         arg_types=[mha_in_ty, qkv_ty], 
    #                         include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
    #                         ],
    #                ) for i in range (4)]
    # k_kernels = [ExternalFunction(f"k1_head{i}",
    #                         source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_k_head{i}.cc"),
    #                         arg_types=[mha_in_ty, qkv_ty], 
    #                         include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
    #                         ],
    #                ) for i in range (4)]
    # v_kernels = [ExternalFunction(f"v1_head{i}",
    #                         source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_v_head{i}.cc"),
    #                         arg_types=[mha_in_ty, qkv_ty], 
    #                         include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
    #                         ],
    #                ) for i in range (4)]

    score_kernels = [ExternalFunction(f"scores1_head{i}",
                            source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_scores_head{i}.cc"),
                            arg_types=[qkv_ty, qkv_ty, score_ty], 
                            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
                            ],
                   ) for i in range (4)]
    
    context_kernels = [ExternalFunction(f"context1_head{i}",
                            source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_context_head{i}.cc"),
                            arg_types=[score_ty, qkv_ty, context_ty], 
                            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
                            ],
                   ) for i in range (4)]
    
    concat_kernels = [ExternalFunction(f"concat1_{i}",
                            source_file = os.path.join(os.path.dirname(__file__), f"iron_kernels/layer_1_concat.cc"),
                            arg_types=[context_ty, context_ty, concat_ty], 
                            include_dirs=[cxx_header_path(), os.path.join(os.path.dirname(__file__), "iron_kernels")
                            ],
                   ) for i in range (2)]

    out_kernel = ExternalFunction(
        "out1",
        source_file=os.path.join(os.path.dirname(__file__), "iron_kernels/layer_1_out.cc"),
        arg_types=[concat_ty, concat_ty, out_ty],
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

    def core_body_qkv (of_in, of_out0, of_out1, of_out2, qkv_kernel):
        elem_in = of_in.acquire(1)
        elem_out0 = of_out0.acquire(1)
        elem_out1 = of_out1.acquire(1)
        elem_out2 = of_out2.acquire(1)
        qkv_kernel(elem_in, elem_out0, elem_out1, elem_out2)
        of_in.release(1)
        of_out0.release(1)
        of_out1.release(1)
        of_out2.release(1)

    workers = []
    workers.append(Worker(core_body_dense, fn_args=[of_in.cons(), of_mha_in.prod(), dense_ly_kernel]))
    for i in range(4):
        workers.append(Worker(core_body_qkv, fn_args=[of_mha_in.cons(), 
                                                        of_q_out[i].prod(), of_k_out[i].prod(), of_v_out[i].prod(),
                                                        qkv_kernels[i]]))
        workers.append(Worker(core_body_in2, fn_args=[of_q_out[i].cons(), of_k_out[i].cons(), of_score[i].prod(), score_kernels[i]]))
        workers.append(Worker(core_body_in2, fn_args=[of_score[i].cons(), of_v_out[i].cons(), of_context[i].prod(), context_kernels[i]]))

    workers.append(Worker(core_body_in2, fn_args=[of_context[0].cons(), of_context[1].cons(), of_concat[0].prod(), concat_kernels[0]]))
    workers.append(Worker(core_body_in2, fn_args=[of_context[2].cons(), of_context[3].cons(), of_concat[1].prod(), concat_kernels[1]]))
    workers.append(Worker(core_body_in2, fn_args=[of_concat[0].cons(), of_concat[1].cons(), of_out.prod(), out_kernel]))


    # for i in range(2):
    #     workers.append(Worker(core_body_qkv, fn_args=[of_mha_in.cons(), 
    #                                                     of_q_out[i].prod(), of_k_out[i].prod(), of_v_out[i].prod(),
    #                                                     qkv_kernels[i]]))
    #     workers.append(Worker(core_body_in2, fn_args=[of_q_out[i].cons(), of_k_out[i].cons(), of_score[i].prod(), score_kernels[i]]))
    #     workers.append(Worker(core_body_in2, fn_args=[of_score[i].cons(), of_v_out[i].cons(), of_context[i].prod(), context_kernels[i]]))

    # workers.append(Worker(core_body_in2, fn_args=[of_context[0].cons(), of_context[1].cons(), of_concat[0].prod(), concat_kernels[0]]))
    # workers.append(Worker(core_body_in2, fn_args=[of_concat[0].cons(), of_concat[0].cons(), of_out.prod(), out_kernel]))
    
    # Runtime and data movement
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_x, c_z):
        rt.start(*workers)
        rt.fill(of_in.prod(), a_x)
        rt.drain(of_out.cons(), c_z, wait=True)

    # Program + placement
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

    # Convert/set Iron tensors for kernel input and output
    inp_tensor = iron.tensor(inp_tiled, dtype=np.int8, device="npu")
    output = iron.zeros(OUTPUT_SIZE, dtype=element_type, device="npu")

    # Insantiate AIE Kernel
    particle_transformer(inp_tensor, output)

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
