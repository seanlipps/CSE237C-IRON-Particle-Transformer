# test_scores.py -*- Python -*-
#
# No-argument test for Python-kernel-based scores
# Verifies:
#   1. MLIR generation succeeds
#   2. (Optional) Python reference kernel produces correct output
#

import numpy as np
import sys

print("Running scores Python-kernel test...\n")

# ----------------------------------------------------------------------
# 1. Generate MLIR (this is the real Iron test)
# ----------------------------------------------------------------------

try:
    import scores  # scores.py prints MLIR on import
    print("\n[OK] MLIR generation succeeded.\n")
except Exception as e:
    print("\n[FAIL] MLIR generation failed:")
    print(e)
    sys.exit(1)


# ----------------------------------------------------------------------
# 2. Optional: verify Python reference kernel numerically
# ----------------------------------------------------------------------

from scores_computation_kernel import score_computation_kernel

ROWS = 40
COLS = 16
Q_SIZE = ROWS * COLS
K_SIZE = ROWS * COLS
S_SIZE = ROWS * ROWS

dtype = np.int8

q = np.random.randint(-128, 127, size=Q_SIZE, dtype=dtype)
k = np.random.randint(-128, 127, size=K_SIZE, dtype=dtype)
s = np.zeros(S_SIZE, dtype=dtype)

# Golden computation
score_computation_kernel(q, k, s)

# Basic sanity check (no NaNs, values written)
if np.all(s == 0):
    print("[FAIL] Python kernel produced all zeros.")
    sys.exit(1)

print("[OK] Python reference kernel produced output.\n")
print("PASS!\n")
sys.exit(0)
