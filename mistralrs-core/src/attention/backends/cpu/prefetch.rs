#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_mm_prefetch;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) unsafe fn prefetch(addr: *const u8) {
    unsafe {
        _mm_prefetch(addr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn prefetch(addr: *const u8) {
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) addr,
            options(readonly, nostack, preserves_flags)
        );
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
pub(super) unsafe fn prefetch(_addr: *const u8) {}
