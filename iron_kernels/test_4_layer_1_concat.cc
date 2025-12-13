#include "iron_kernels.h"

extern "C" {
// Concat head 0 and head 1: (40,16) + (40,16) -> (40,32)
void concat1_0(int8_t * sA, int8_t * sB, int8_t * sC){
    concat<4, 8, 10, 2>(sA, sB, sC);
}

// Concat head 2 and head 3: (40,16) + (40,16) -> (40,32)
void concat1_1(int8_t * sA, int8_t * sB, int8_t * sC){
    concat<4, 8, 10, 2>(sA, sB, sC);
}
}