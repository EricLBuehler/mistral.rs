use rayon::ThreadPool;
use std::sync::LazyLock;

pub(super) static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(candle_core::utils::get_num_threads())
        .start_handler(|_| candle_core::utils::set_thread_affinity())
        .build()
        .expect("Failed to build custom Rayon thread-pool for flash-attention")
});
