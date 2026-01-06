#include "iron_kernels.h"

extern "C" {
#include <cstdint>

void f3(int8_t * x1, int8_t * x2, int8_t * a){ 
    resadd<4, 8, 40, 8>(x1, x2, a);
}
}