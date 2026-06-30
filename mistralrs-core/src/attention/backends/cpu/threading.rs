use rayon::ThreadPool;
use std::sync::LazyLock;

#[cfg(target_os = "macos")]
unsafe fn set_thread_affinity() {
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
#[inline(always)]
unsafe fn set_thread_affinity() {}

pub(super) static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            set_thread_affinity();
        })
        .build()
        .expect("Failed to build custom Rayon thread-pool for flash-attention")
});
