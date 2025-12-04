#include "iron_kernels.h"

extern "C" {
#include <cstdint>

void f0(int8_t * x1, int8_t * x2, int8_t * a){ 
    concat<4, 8, 40, 2>(x1, x2, a);
}
}