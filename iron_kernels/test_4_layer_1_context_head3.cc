#include "iron_kernels.h"

extern "C" {
void context1_head3(int8_t * x, int8_t * v, int8_t * a){
    context<4, 8, 8, 10, 5, 2, 10>(x, v, a);
}
}