use rayon::ThreadPool;
use std::sync::LazyLock;

pub(super) static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(mistralrs_quant::cpu_threads::resolve())
        .start_handler(|_| candle_core::utils::set_thread_affinity())
        .build()
        .expect("Failed to build custom Rayon thread-pool for flash-attention")
});
