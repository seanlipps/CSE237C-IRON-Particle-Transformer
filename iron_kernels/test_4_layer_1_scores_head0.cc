#include "iron_kernels.h"

extern "C" {
#include <cstdint>

void score_kernel(int8_t * x1, int8_t * x2, int8_t * a){ 
    scores<4, 8, 8, 10, 8, 8, 64, 40, 8>(x1, x2, a);
}
}