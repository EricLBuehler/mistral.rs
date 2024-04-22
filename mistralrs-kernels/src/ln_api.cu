#include "ln.h"
#include "ln_fwd_kernels.cuh"

/*
Ada

Supported Type combinations:

input  residual   compute   weights   output
============================================
fp32     fp32      fp32      fp32      fp32
fp16     fp32      fp32      fp32      fp16
fp16     fp16      fp32      fp32      fp16
bf16     fp32      fp32      fp32      bf16
bf16     bf16      fp32      fp32      bf16
fp16     fp16      fp32      fp16      fp16
bf16     bf16      fp32      bf16      bf16

Remarks:
Output type = Input type
Compute always in FP32

*/

namespace layer_norm {

FwdRegistry FWD_FUNCS;

uint64_t get_key(uint32_t wtype, uint32_t itype, uint32_t rtype, uint32_t otype, uint32_t ctype, uint64_t hidden_size) {
    using namespace layer_norm;
    uint64_t type_key = wtype | (itype << 2) | (rtype << 4) | (otype << 6) | (ctype << 8);
    uint64_t launcher_key = (type_key << 32) | hidden_size;
    return launcher_key;
}

}

layer_norm::FwdFunction & get_fwd_launcher(uint32_t wtype, uint32_t itype, uint32_t rtype, uint32_t otype, uint32_t ctype, uint32_t hidden_size) {
    auto iter = layer_norm::FWD_FUNCS.find(layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    return iter->second;
}

REGISTER_FWD_LAUNCHER(  256, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  256, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER(  512, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  512, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER(  768, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER(  768, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER( 1024, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1024, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER( 1280, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1280, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER( 1536, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 1536, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER( 2048, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2048, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER( 2560, fp32, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp16, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp32, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp16, fp16, fp32, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp32, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp32, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, bf16, bf16, fp32, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp32, bf16, bf16, bf16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, fp16, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_LAUNCHER( 2560, bf16, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_LAUNCHER( 3072, fp32, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp16, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp32, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp16, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp32, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp32, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, bf16, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp32, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, fp16, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 3072, bf16, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER( 4096, fp32, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp16, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp16, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, bf16, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp32, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, fp16, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 4096, bf16, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER( 5120, fp32, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp16, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp32, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp16, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp32, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp32, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, bf16, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp32, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, fp16, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 5120, bf16, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER( 6144, fp32, fp32, fp32, fp32, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp16, fp32, fp32, fp32, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp32, fp16, fp32, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp16, fp16, fp32, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp32, fp16, fp16, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp32, bf16, fp32, bf16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, bf16, bf16, fp32, bf16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp32, bf16, bf16, bf16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, fp16, fp16, fp16, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 6144, bf16, bf16, bf16, bf16, fp32, 1, 1, 8, 16);

REGISTER_FWD_LAUNCHER( 7168, fp32, fp32, fp32, fp32, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 7168, fp16, fp32, fp32, fp32, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 7168, fp32, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, fp16, fp16, fp32, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, fp32, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, fp32, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, bf16, bf16, fp32, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, fp32, bf16, bf16, bf16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, fp16, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_LAUNCHER( 7168, bf16, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

REGISTER_FWD_LAUNCHER( 8192, fp32, fp32, fp32, fp32, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp16, fp32, fp32, fp32, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp32, fp16, fp32, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp16, fp16, fp32, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp32, fp16, fp16, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp32, bf16, fp32, bf16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, bf16, bf16, fp32, bf16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp32, bf16, bf16, bf16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, fp16, fp16, fp16, fp16, fp32, 1, 1, 8, 16);
REGISTER_FWD_LAUNCHER( 8192, bf16, bf16, bf16, bf16, fp32, 1, 1, 8, 16);

extern "C" void run_ln(
    void *x,
    void *residual,
    void *gamma,
    void *beta,
    void *dst_add,
    void *dst,
    void *mu,
    void *rsigma,

    float epsilon,

    uint32_t hidden_size_rounded,
    uint32_t rows,
    uint32_t cols,
    int32_t multi_processor_count,

    uint32_t wtype,
    uint32_t itype,
    uint32_t rtype,
    uint32_t otype,
    uint32_t ctype,

    int is_rms_norm
) {
    layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

    launch_params.multi_processor_count = multi_processor_count;
    launch_params.stream = 0;

    launch_params.params.dropout_keep_p = 1.f;
    launch_params.params.residual = residual;
    launch_params.params.rowscale = nullptr;
    launch_params.params.colscale = nullptr;
    launch_params.params.x0_subset = nullptr;
    launch_params.params.z_subset = nullptr;

    // Request the kernel launcher.
    auto launcher = get_fwd_launcher(wtype, itype, rtype, otype, ctype, hidden_size_rounded);

    // Set the kernel runtime parameters.
    layer_norm::FwdParams &params = launch_params.params;

    params.rows = rows;
    params.cols = cols;
    params.x0 = x;
    params.x = dst_add;
    params.dmask = nullptr;
    params.mu = mu;
    params.rs = rsigma;
    params.gamma = gamma;
    params.beta = beta;
    params.z = dst;
    params.epsilon = epsilon;
    params.dropout_scale = 1.f;
    params.inverse_cols = 1.f / float(params.cols);
    params.rowscale_const = 1.f;
    params.is_rms_norm = is_rms_norm;

    // Query the kernel-specific launch parameters.
    launcher(launch_params, true);

    // Launch the kernel.
    launcher(launch_params, false);
}