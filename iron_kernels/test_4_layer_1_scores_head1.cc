#include "iron_kernels.h"

extern "C" {
void scores1_head1(int8_t * q_head, int8_t * k_head, int8_t * o_head){
    scores<4, 8, 8, 10, 2, 2, 16, 40, 8>(q_head, k_head, o_head);
}
}