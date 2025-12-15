# scores.py -*- Python -*-
#
# Python-kernel-based scores computation:
#   S = Q * K^T
#

import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.dialects.ext.func import func


# --------------------------------------------------------------------------
# Device selection (same pattern as passthrough)
# --------------------------------------------------------------------------

dev = NPU1Col1()

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"[ERROR] Device name {sys.argv[1]} is unknown")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

ROWS = 40
COLS = 16
SHIFT_S = 8

Q_SIZE = ROWS * COLS
K_SIZE = ROWS * COLS
S_SIZE = ROWS * ROWS

dtype = np.int8

q_type = np.ndarray[(Q_SIZE,), np.dtype[dtype]]
k_type = np.ndarray[(K_SIZE,), np.dtype[dtype]]
s_type = np.ndarray[(S_SIZE,), np.dtype[dtype]]


# --------------------------------------------------------------------------
# ObjectFifos
# --------------------------------------------------------------------------

of_q = ObjectFifo(q_type, name="q_in")
of_k = ObjectFifo(k_type, name="k_in")
of_s = ObjectFifo(s_type, name="s_out")


# --------------------------------------------------------------------------
# Python kernel (THIS is the important part)
# --------------------------------------------------------------------------

@func
def score_computation_kernel(
    q_in: q_type,
    k_in: k_type,
    s_out: s_type,
):
    shift = 8
    for r in range_(ROWS):
        for c in range_(ROWS):
            acc = 0
            for k in range_(COLS):
                acc += q_in[r * COLS + k] * k_in[c * COLS + k]
            s_out[c] = acc



# --------------------------------------------------------------------------
# Core task
# --------------------------------------------------------------------------

def core_fn(of_q, of_k, of_s, kernel):
    q = of_q.acquire(1)
    k = of_k.acquire(1)
    s = of_s.acquire(1)

    kernel(q, k, s)

    of_q.release(1)
    of_k.release(1)
    of_s.release(1)


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

worker = Worker(
    core_fn,
    [
        of_q.cons(),
        of_k.cons(),
        of_s.prod(),
        score_computation_kernel,
    ],
)


# --------------------------------------------------------------------------
# Runtime sequence (symbolic buffers)
# --------------------------------------------------------------------------

rt = Runtime()
with rt.sequence(q_type, k_type, s_type) as (a_q, a_k, a_s):
    rt.start(worker)
    rt.fill(of_q.prod(), a_q)
    rt.fill(of_k.prod(), a_k)
    rt.drain(of_s.cons(), a_s, wait=True)


# --------------------------------------------------------------------------
# Program → MLIR
# --------------------------------------------------------------------------

program = Program(dev, rt)
module = program.resolve_program(SequentialPlacer())

print(module)
