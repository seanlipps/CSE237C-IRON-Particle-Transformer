#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>


extern "C" {
// (Q @ K^T):  (T, d_model) @ (T, d_model)^T -> (T, T)
// m=4, k=8, n=8, T=160, d_model=64, Tm(rows)=160/m=40, Tn(columns)=64/n=8
//template <int m, int k, int n, int Tm, int Tk, int Tn, int d_model, int T, int SHIFT_S>
void scores(
  int8_t * __restrict pQ, // adf::input_buffer<int8, adf::extents<T*d_model>> & sQ,
  int8_t * __restrict pK, // adf::input_buffer<int8, adf::extents<T*d_model>> & sK,
  int8_t * __restrict pS
) {
//  using MMUL = aie::mmul<m, n, m, int8, int8>; // 4x8x4
//  using VA   = aie::vector<int8, MMUL::size_A>; // 4x8
//  using VB   = aie::vector<int8, MMUL::size_A>; // 8x4
//  using VC   = aie::vector<int8, MMUL::size_C>; // 4x4
//
//  const int8_t* ptrQ = pQ;
//  const int8_t* ptrK = pK;
//  int8_t* ptrS = pS;
//  VB matB[Tm*Tn]; //store all of pK in mem

  int8_t* a = pQ;
  int8_t* b = pK;
  int8_t* c = pS;

  // Iron flattens things out in a pointer. So assuming we transposed correctly I'll just copy element by element
  for(int i = 0; i < 32; ++i){
    c[i] = a[i];
  }

  // going to assume we need a 4 by 4 kernel as output
  //for(int i = 0; i < 4; ++i){
  //  for(int j = 0; j < 8; ++j){
  //    // row*row_length + column
  //    c[i*(4)+j] = a[i*4+j];
  //  }
  //}

//  for (unsigned i = 0; i < Tm; ++i) { // rows
//    for (unsigned j = 0; j < Tn; ++j) { // columns
//      matB[i*Tn+j] = aie::transpose(aie::load_v<MMUL::size_A>(ptrK), m, n);
//      ptrK += MMUL::size_A;
//    }
//  }
//  
//  // row by row multiplication
//  for (unsigned im = 0; im < Tm; ++im) {   // rows of Q
//    VA Abuf[Tn]; // row of tiles
//    for (unsigned in = 0; in < Tn; ++in) { // columns of Q
//      Abuf[in] = aie::load_v<MMUL::size_A>(ptrQ);
//      ptrQ += MMUL::size_A;
//    }
//    for (unsigned jm = 0; jm < Tm; ++jm) { // rows of K
//      MMUL C;
//      for (unsigned in = 0; in < Tn; ++in) { // columns of K
//        if (in == 0) C.mul(Abuf[0], matB[jm*Tn+in]);
//        else         C.mac(Abuf[in], matB[jm*Tn+in]);
//      }
//      VC V = C.template to_vector<int8>(SHIFT_S);
//      aie::store_v(ptrS, v);
//      ptrS += MMUL::size_C;
//    }
//  }
}
}
