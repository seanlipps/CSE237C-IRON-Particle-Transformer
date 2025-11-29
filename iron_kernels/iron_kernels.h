#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense_kernel(
int8_t * __restrict pA,
int8_t * __restrict pC,
const int8_t * matB
) {
    using MMUL = aie::mmul<m, k, n, int8, int8>;
    using VA   = aie::vector<int8, MMUL::size_A>;
    using VB   = aie::vector<int8, MMUL::size_B>;
    using VC   = aie::vector<int8, MMUL::size_C>;
    
    const int8_t* __restrict Bbase = matB;
    const unsigned strideB_perK  = MMUL::size_B * Tn;
    
    const int8_t* ptrA = pA;
    int8_t* ptrC = pC;
    
    for (unsigned im = 0; im < Tm; ++im) {
        VA Abuf[Tk];
        for (unsigned ik = 0; ik < Tk; ++ik){
            Abuf[ik] = aie::load_v<MMUL::size_A>(ptrA);
            ptrA += MMUL::size_A;
        }
        for (unsigned in = 0; in < Tn; ++in) {
            MMUL C;
            const int8_t* __restrict pB = Bbase + in * MMUL::size_B;
            for (unsigned ik = 0; ik < Tk; ++ik) {
                VB b = aie::load_v<MMUL::size_B>(pB + ik * strideB_perK);
                if (ik == 0) C.mul(Abuf[0], b);
                else         C.mac(Abuf[ik], b);
            }
        
            VC v = C.template to_vector<int8>(SHIFT);
            if (is_relu) v = aie::max(v, (int8)0);
            
            aie::store_v(ptrC, v);
            ptrC += MMUL::size_C;
        }
    }
}