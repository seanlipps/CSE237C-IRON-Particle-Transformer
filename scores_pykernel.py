import numpy as np

def score_computation_kernel(q_in: np.ndarray, k_in: np.ndarray, s_out: np.ndarray):
    """
    Computes S = Q * K^T, where Q and K are 1D flattened arrays.

    The dimensions are hardcoded to match the test data:
    Q: 40x64 (flattened to 2560)
    K: 40x64 (flattened to 2560)
    S: 40x40 (flattened to 1600)
    """
    
    # Dimensions from your test_4_mha_score_only.py
    ROWS = 40  # M and N in a M x K * K x N multiplication
    COLS = 16  # K dimension
    SHIFT_S = 8 # Mimicking the shift used in the C++ kernel

    # Q is ROWS x COLS, K is ROWS x COLS, S is ROWS x ROWS
    
    for r in range(ROWS):
        for c in range(ROWS):
            # Calculate S[r][c] = dot_product(Q[r,:], K[c,:])
            # K is transposed implicitly by iterating over its rows (c) 
            # and using the same inner loop index (k) for Q and K.
            acc = np.int32(0) # Use a larger accumulator type
            
            for k in range(COLS):
                # Indices for the flattened 1D arrays
                idx_q = r * COLS + k
                idx_k = c * COLS + k
                
                # Multiply and accumulate
                acc += np.int32(q_in[idx_q]) * np.int32(k_in[idx_k])

            # Apply bit-shift (scaling) and cast to output type (np.int8)
            # This is critical for fixed-point quantization used in AIE.
            res = (acc >> SHIFT_S)
            
            # Store result in the flattened output array
            s_out[r * ROWS + c] = np.int8(res)