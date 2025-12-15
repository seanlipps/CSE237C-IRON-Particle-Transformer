#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include <vector>

template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense(
int8_t * __restrict pA,
int8_t * __restrict pC,
const int8_t * matB
) {
    using MMUL = aie::mmul<m, k, n, int8, int8>; // 4x8x8
    using VA   = aie::vector<int8, MMUL::size_A>; // 4x8
    using VB   = aie::vector<int8, MMUL::size_B>; // 8x8
    using VC   = aie::vector<int8, MMUL::size_C>; // 4x8
    
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


// (Q @ K^T):  (T, head_dim) @ (T, head_dim)^T -> (T, T)
// 160*16 @ 160*16^T = 160*160
// m=4, k=8, n=8, T=160, d_model=64, head_dim = 64/4 = 16, Tm(rows)=160/m=40, Tk = head_dim/k = 16/8 = 2, Tn (columns)= head_dim/k = 16/8 = 2
template <int m, int k, int n, int Tm, int Tk, int Tn, int d_model, int T, int SHIFT_S>
void scores(
  int8_t * __restrict pQ,
  int8_t * __restrict pK,
  int8_t * __restrict pS
) {
  using MMUL = aie::mmul<m, n, n, int8, int8>; // 4x8x8
  using VA   = aie::vector<int8, MMUL::size_A>; // 4x8
  using VB   = aie::vector<int8, MMUL::size_B>; // 8x8
  using VC   = aie::vector<int8, MMUL::size_C>; // 4x8

  using VCout= aie::vector<int8, m*m>; // 4x4

  const int8_t* ptrQ = pQ;
  const int8_t* ptrK = pK;
  int8_t* ptrS = pS;
    
  alignas(64) static VB matB[Tm*Tn]; //store all of pK in mem

  alignas(32) int8_t tile[m*n]; //4x8
  alignas(32) int8_t trans_tile[n*n] = {}; //8x8, initialize with 0s

  alignas(32) int8_t otile[MMUL::size_C]; //4x8
  alignas(32) int8_t out_tile[m*m]; //4x4

  unsigned d = 0;
  for (unsigned i = 0; i < Tm; ++i) { // rows
    for (unsigned j = 0; j < Tn; ++j) { // columns
      VA Kt = aie::transpose(aie::load_v<MMUL::size_A>(ptrK), m, n);
      ptrK += MMUL::size_A;
      aie::store_v(tile, Kt);
      unsigned e = 0;
      for (unsigned r = 0; r < n; ++r) {
          for (unsigned c = 0; c < m; ++c) {
              trans_tile[r*n+c] = tile[e++];
          }
      }
      matB[d++] = aie::load_v<n*n>(trans_tile);
    }
  }
    
  // row by row multiplication
  for (unsigned im = 0; im < Tm; ++im) {
    alignas(64) VA Abuf[Tn]; // row of tiles
    for (unsigned in = 0; in < Tn; ++in) {
      Abuf[in] = aie::load_v<MMUL::size_A>(ptrQ);
      ptrQ += MMUL::size_A;
    }
    unsigned d = 0;
    for (unsigned jm = 0; jm < Tm; ++jm) { // rows of K
      MMUL C;
      for (unsigned in = 0; in < Tn; ++in) { // columns of K
        if (in == 0) C.mul(Abuf[0], matB[d++]);
        else         C.mac(Abuf[in], matB[d++]);
      }
      VC v = C.template to_vector<int8>(SHIFT_S); //4x8
      aie::store_v(otile, v);
      for (unsigned r = 0; r < m; ++r) {
          for (unsigned c = 0; c < m; ++c) {
              out_tile[r*m+c] = otile[r*n+c];
          }
      }
      VCout vout= aie::load_v<m*m>(out_tile);//4x4
      aie::store_v(ptrS, vout);
      ptrS += m*m; 
    }
  }
}


// (scores @ V)  (T,T) @ (T,head_dim) -> (T,head_dim)
// Tm = 160/4 = 40, Tk = 160/8 = 20, Tn = 16/8 = 2
// 160 x 160 x 16 tiled with 4 x 8 x 8
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT>
void context(
  int8_t * __restrict pS,
  int8_t * __restrict pV,
  int8_t * __restrict pC
) {
  using MMUL = aie::mmul<m, m, n, int8, int16>; // 4x4x8 -> 4x8
  using VA   = aie::vector<int8,  MMUL::size_A>; // 4x4 (int8)
  using VB   = aie::vector<int16, MMUL::size_B>; // 4x8 (int16)
  using VC   = aie::vector<int16, MMUL::size_C>; // 4x8 (int16)

  using VBin = aie::vector<int8, m*n>; // 4x8 (int8)
  using VCout = aie::vector<int8, m*n>; // 4x8 (int8)

  const int8_t* ptrS = pS;
  const int8_t* ptrV = pV;
  int8_t* ptrC = pC;
  static VB matB[Tm*Tn];

  for (unsigned im = 0; im < Tm; ++im) { // rows
    for (unsigned in = 0; in < Tn; ++in) { // columns
      VBin B = aie::load_v<m*n>(ptrV); // 4x8
      ptrV += m*n;
      VB B16 = B.unpack();
      matB[im*Tn+in] = B16; //convert to int16 for 4x4x8
    }
  }

  for (unsigned im = 0; im < Tm; ++im) {
  // chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tm];
    for (unsigned jm = 0; jm < Tm; ++jm) {
      Abuf[jm] = aie::load_v<MMUL::size_A>(ptrS); // one tile
      ptrS += MMUL::size_A;
    }
    for (unsigned in = 0; in < Tn; ++in) {
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      for (unsigned jm = 0; jm < Tm; ++jm) {//row of B
        if (jm == 0) C.mul(Abuf[0], matB[jm*Tn+in]);
        else         C.mac(Abuf[jm], matB[jm*Tn+in]);
      }

      VC v = C.template to_vector<int16>(SHIFT);
      VCout vout = v.pack();
      aie::store_v(ptrC, vout);
      ptrC += m*n;
    }
  }
}


// Concatenate two (m*Tm) x (n*Tn) int8 matrices.
template <int m, int n, int Tm, int Tn>
void concat(
  int8_t * __restrict pA,
  int8_t * __restrict pB,
  int8_t * __restrict pC
) {
  using V = aie::vector<int8, m*n>;
 
  const int8_t* ptrA = pA;
  const int8_t* ptrB = pB;
  int8_t* ptrC = pC;

  for (int im = 0; im < Tm; ++im) {
    for (int in = 0; in < Tn; ++in) {
      aie::store_v(ptrC, aie::load_v<m*n>(ptrA));
      ptrA += m*n;
      ptrC += m*n;
    }
    for (int in = 0; in < Tn; ++in) {
      aie::store_v(ptrC, aie::load_v<m*n>(ptrB));
      ptrB += m*n;
      ptrC += m*n;
    }
  }
}


// (context @ Wo)  (T,d_model) @ (d_model,d_model) -> (T,d_model)
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT_O>
void output(
  int8_t* __restrict pA,
  int8_t* __restrict pB,
  int8_t* __restrict pO,
  const int8_t Wo[]
) {
  using MMUL = aie::mmul<m, k, n, int8, int8>;
  using VA   = aie::vector<int8, MMUL::size_A>;
  using VB   = aie::vector<int8, MMUL::size_B>;
  using VC   = aie::vector<int8, MMUL::size_C>;

  const int8_t* ptrA = pA;
  const int8_t* ptrB = pB;
  int8_t* ptrO = pO;

  const int8* __restrict Bbase = (const int8*)Wo;
  const unsigned strideB_perK  = MMUL::size_B * Tn;

  for (unsigned im = 0; im < Tm; ++im) {
  // chess_prepare_for_pipelining chess_loop_range(1,) {
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk/2; ++ik) {
      Abuf[ik] = aie::load_v<MMUL::size_A>(ptrA);  
      ptrA += MMUL::size_A;
    }
    for (unsigned ik = Tk/2; ik < Tk; ++ik) {
      Abuf[ik] = aie::load_v<MMUL::size_A>(ptrB);
      ptrB += MMUL::size_A;
    }

    for (unsigned in = 0; in < Tn; ++in) {
    // chess_prepare_for_pipelining chess_loop_range(1,) {
      MMUL C;
      const int8* __restrict pcol = Bbase + in * MMUL::size_B; // pcol points to the starting address of each tile 

      for (unsigned ik = 0; ik < Tk; ++ik) {
        VB b = aie::load_v<MMUL::size_B>(pcol + ik * strideB_perK); 
        if (ik == 0) C.mul(Abuf[0], b);
        else         C.mac(Abuf[ik], b);
      }

      VC v = C.template to_vector<int8>(SHIFT_O);
      v = aie::max(v, (int8)0);
      aie::store_v(ptrO, v);
      ptrO += MMUL::size_C;
    }
  }
}


// Add two (m*Tm) x (n*Tn) int8 matrices element-wise with saturation.
template <int m, int n, int Tm, int Tn>
void resadd(
    int8_t * __restrict sA,
    int8_t * __restrict sB,
    int8_t * __restrict sC
) {
    using V = aie::vector<int8, m*n>;

    // Explicitly set saturation mode
    aie::set_saturation(aie::saturation_mode::saturate);
    
    const int8_t* ptrA = sA;
    const int8_t* ptrB = sB;
    int8_t* ptrC = sC;
    
    for (int im = 0; im < Tm; ++im) {
        for (int in = 0; in < Tn; ++in) {
            V vA = aie::load_v<m*n>(ptrA);
            V vB = aie::load_v<m*n>(ptrB);
            V vC = aie::saturating_add(vA, vB);  // saturating addition
            aie::store_v(ptrC, vC);
            ptrA += m*n;
            ptrB += m*n;
            ptrC += m*n;
        }
    }
}