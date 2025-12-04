#include "iron_kernels.h"

extern "C" {
#include <cstdint>

void f0(int8_t * x1, int8_t * x2, int8_t * a){ 
    context<4, 8, 8, 40, 20, 2, 10>(x1, x2, a);
}
}