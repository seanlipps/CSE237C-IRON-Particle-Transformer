#include "iron_kernels.h"

extern "C" {
#include <cstdint>

void context_kernel(int8_t * x1, int8_t * x2, int8_t * a){ 
    context<4, 8, 8, 10, 5, 8, 9>(x1, x2, a);
}
}