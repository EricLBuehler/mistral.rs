// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_splitkv_paged_<cutlass::bfloat16_t, 512, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_splitkv_paged_hdim512<cutlass::bfloat16_t, true>(params, stream);
}
