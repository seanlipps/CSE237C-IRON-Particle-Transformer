#include "iron_kernels.h"

extern "C" {
void context1_head1(int8_t * x, int8_t * v, int8_t * a){
    context<4, 8, 8, 10, 5, 2, 9>(x, v, a);
}
}