//! FFI bindings for AFQ CUDA kernels.
//!
//! These bindings allow Rust code to call the optimized CUDA kernels
//! for AFQ quantization, dequantization, and matrix operations.

#![allow(dead_code)]
#![allow(improper_ctypes)]

use half::{bf16, f16};

// ============================================================================
// Dequantize kernel bindings
// ============================================================================

macro_rules! dequant_kernel_power_of_2 {
    ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_dequantize_ $bits bit_gs $gs _ $postfix >](
                w_q: *const u32,
                scales: *const $scalar,
                biases: *const $scalar,
                output: *mut $scalar,
                rows: i32,
                cols: i32,
            );
        }
    };
}

macro_rules! dequant_kernel_3bit {
    ($gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_dequantize_3bit_gs $gs _ $postfix >](
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                output: *mut $scalar,
                rows: i32,
                cols: i32,
            );
        }
    };
}

macro_rules! dequant_kernel_6bit {
    ($gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_dequantize_6bit_gs $gs _ $postfix >](
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                output: *mut $scalar,
                rows: i32,
                cols: i32,
            );
        }
    };
}

// ============================================================================
// Quantize kernel bindings
// ============================================================================

macro_rules! quant_kernel {
    ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_quantize_ $bits bit_gs $gs _ $postfix >](
                w: *const $scalar,
                w_q: *mut u32,
                scales: *mut $scalar,
                biases: *mut $scalar,
                rows: i32,
                cols: i32,
            );
        }
    };
}

// ============================================================================
// QMV (quantized matrix-vector) kernel bindings
// ============================================================================

macro_rules! qmv_kernel_power_of_2 {
    ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_qmv_ $bits bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u32,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
            );
        }
    };
}

macro_rules! qmv_kernel_3bit {
    ($gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_qmv_3bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
            );
        }
    };
}

macro_rules! qmv_kernel_6bit {
    ($gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_qmv_6bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u8,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
            );
        }
    };
}

// ============================================================================
// QMM (quantized matrix-matrix) kernel bindings
// ============================================================================

macro_rules! qmm_kernel {
    ($bits:tt, $gs:tt, $scalar:ty, $postfix:tt) => {
        paste::paste! {
            pub fn [< afq_qmm_ $bits bit_gs $gs _ $postfix >](
                x: *const $scalar,
                w_q: *const u32,
                scales: *const $scalar,
                biases: *const $scalar,
                y: *mut $scalar,
                m: i32,
                n: i32,
                k: i32,
            );
        }
    };
}

// ============================================================================
// Extern "C" declarations
// ============================================================================

extern "C" {
    // --- Dequantize: 2-bit ---
    dequant_kernel_power_of_2!(2, 32, f32, f32);
    dequant_kernel_power_of_2!(2, 64, f32, f32);
    dequant_kernel_power_of_2!(2, 128, f32, f32);
    dequant_kernel_power_of_2!(2, 32, f16, f16);
    dequant_kernel_power_of_2!(2, 64, f16, f16);
    dequant_kernel_power_of_2!(2, 128, f16, f16);
    dequant_kernel_power_of_2!(2, 32, bf16, bf16);
    dequant_kernel_power_of_2!(2, 64, bf16, bf16);
    dequant_kernel_power_of_2!(2, 128, bf16, bf16);

    // --- Dequantize: 3-bit ---
    dequant_kernel_3bit!(32, f32, f32);
    dequant_kernel_3bit!(64, f32, f32);
    dequant_kernel_3bit!(128, f32, f32);
    dequant_kernel_3bit!(32, f16, f16);
    dequant_kernel_3bit!(64, f16, f16);
    dequant_kernel_3bit!(128, f16, f16);
    dequant_kernel_3bit!(32, bf16, bf16);
    dequant_kernel_3bit!(64, bf16, bf16);
    dequant_kernel_3bit!(128, bf16, bf16);

    // --- Dequantize: 4-bit ---
    dequant_kernel_power_of_2!(4, 32, f32, f32);
    dequant_kernel_power_of_2!(4, 64, f32, f32);
    dequant_kernel_power_of_2!(4, 128, f32, f32);
    dequant_kernel_power_of_2!(4, 32, f16, f16);
    dequant_kernel_power_of_2!(4, 64, f16, f16);
    dequant_kernel_power_of_2!(4, 128, f16, f16);
    dequant_kernel_power_of_2!(4, 32, bf16, bf16);
    dequant_kernel_power_of_2!(4, 64, bf16, bf16);
    dequant_kernel_power_of_2!(4, 128, bf16, bf16);

    // --- Dequantize: 6-bit ---
    dequant_kernel_6bit!(32, f32, f32);
    dequant_kernel_6bit!(64, f32, f32);
    dequant_kernel_6bit!(128, f32, f32);
    dequant_kernel_6bit!(32, f16, f16);
    dequant_kernel_6bit!(64, f16, f16);
    dequant_kernel_6bit!(128, f16, f16);
    dequant_kernel_6bit!(32, bf16, bf16);
    dequant_kernel_6bit!(64, bf16, bf16);
    dequant_kernel_6bit!(128, bf16, bf16);

    // --- Dequantize: 8-bit ---
    dequant_kernel_power_of_2!(8, 32, f32, f32);
    dequant_kernel_power_of_2!(8, 64, f32, f32);
    dequant_kernel_power_of_2!(8, 128, f32, f32);
    dequant_kernel_power_of_2!(8, 32, f16, f16);
    dequant_kernel_power_of_2!(8, 64, f16, f16);
    dequant_kernel_power_of_2!(8, 128, f16, f16);
    dequant_kernel_power_of_2!(8, 32, bf16, bf16);
    dequant_kernel_power_of_2!(8, 64, bf16, bf16);
    dequant_kernel_power_of_2!(8, 128, bf16, bf16);

    // --- Quantize: 2-bit ---
    quant_kernel!(2, 32, f32, f32);
    quant_kernel!(2, 64, f32, f32);
    quant_kernel!(2, 128, f32, f32);
    quant_kernel!(2, 32, f16, f16);
    quant_kernel!(2, 64, f16, f16);
    quant_kernel!(2, 128, f16, f16);
    quant_kernel!(2, 32, bf16, bf16);
    quant_kernel!(2, 64, bf16, bf16);
    quant_kernel!(2, 128, bf16, bf16);

    // --- Quantize: 4-bit ---
    quant_kernel!(4, 32, f32, f32);
    quant_kernel!(4, 64, f32, f32);
    quant_kernel!(4, 128, f32, f32);
    quant_kernel!(4, 32, f16, f16);
    quant_kernel!(4, 64, f16, f16);
    quant_kernel!(4, 128, f16, f16);
    quant_kernel!(4, 32, bf16, bf16);
    quant_kernel!(4, 64, bf16, bf16);
    quant_kernel!(4, 128, bf16, bf16);

    // --- Quantize: 8-bit ---
    quant_kernel!(8, 32, f32, f32);
    quant_kernel!(8, 64, f32, f32);
    quant_kernel!(8, 128, f32, f32);
    quant_kernel!(8, 32, f16, f16);
    quant_kernel!(8, 64, f16, f16);
    quant_kernel!(8, 128, f16, f16);
    quant_kernel!(8, 32, bf16, bf16);
    quant_kernel!(8, 64, bf16, bf16);
    quant_kernel!(8, 128, bf16, bf16);

    // --- QMV: 2-bit ---
    qmv_kernel_power_of_2!(2, 32, f32, f32);
    qmv_kernel_power_of_2!(2, 64, f32, f32);
    qmv_kernel_power_of_2!(2, 128, f32, f32);
    qmv_kernel_power_of_2!(2, 32, f16, f16);
    qmv_kernel_power_of_2!(2, 64, f16, f16);
    qmv_kernel_power_of_2!(2, 128, f16, f16);
    qmv_kernel_power_of_2!(2, 32, bf16, bf16);
    qmv_kernel_power_of_2!(2, 64, bf16, bf16);
    qmv_kernel_power_of_2!(2, 128, bf16, bf16);

    // --- QMV: 3-bit ---
    qmv_kernel_3bit!(32, f32, f32);
    qmv_kernel_3bit!(64, f32, f32);
    qmv_kernel_3bit!(128, f32, f32);
    qmv_kernel_3bit!(32, f16, f16);
    qmv_kernel_3bit!(64, f16, f16);
    qmv_kernel_3bit!(128, f16, f16);
    qmv_kernel_3bit!(32, bf16, bf16);
    qmv_kernel_3bit!(64, bf16, bf16);
    qmv_kernel_3bit!(128, bf16, bf16);

    // --- QMV: 4-bit ---
    qmv_kernel_power_of_2!(4, 32, f32, f32);
    qmv_kernel_power_of_2!(4, 64, f32, f32);
    qmv_kernel_power_of_2!(4, 128, f32, f32);
    qmv_kernel_power_of_2!(4, 32, f16, f16);
    qmv_kernel_power_of_2!(4, 64, f16, f16);
    qmv_kernel_power_of_2!(4, 128, f16, f16);
    qmv_kernel_power_of_2!(4, 32, bf16, bf16);
    qmv_kernel_power_of_2!(4, 64, bf16, bf16);
    qmv_kernel_power_of_2!(4, 128, bf16, bf16);

    // --- QMV: 6-bit ---
    qmv_kernel_6bit!(32, f32, f32);
    qmv_kernel_6bit!(64, f32, f32);
    qmv_kernel_6bit!(128, f32, f32);
    qmv_kernel_6bit!(32, f16, f16);
    qmv_kernel_6bit!(64, f16, f16);
    qmv_kernel_6bit!(128, f16, f16);
    qmv_kernel_6bit!(32, bf16, bf16);
    qmv_kernel_6bit!(64, bf16, bf16);
    qmv_kernel_6bit!(128, bf16, bf16);

    // --- QMV: 8-bit ---
    qmv_kernel_power_of_2!(8, 32, f32, f32);
    qmv_kernel_power_of_2!(8, 64, f32, f32);
    qmv_kernel_power_of_2!(8, 128, f32, f32);
    qmv_kernel_power_of_2!(8, 32, f16, f16);
    qmv_kernel_power_of_2!(8, 64, f16, f16);
    qmv_kernel_power_of_2!(8, 128, f16, f16);
    qmv_kernel_power_of_2!(8, 32, bf16, bf16);
    qmv_kernel_power_of_2!(8, 64, bf16, bf16);
    qmv_kernel_power_of_2!(8, 128, bf16, bf16);

    // --- QMM: 2-bit ---
    qmm_kernel!(2, 32, f32, f32);
    qmm_kernel!(2, 64, f32, f32);
    qmm_kernel!(2, 128, f32, f32);
    qmm_kernel!(2, 32, f16, f16);
    qmm_kernel!(2, 64, f16, f16);
    qmm_kernel!(2, 128, f16, f16);
    qmm_kernel!(2, 32, bf16, bf16);
    qmm_kernel!(2, 64, bf16, bf16);
    qmm_kernel!(2, 128, bf16, bf16);

    // --- QMM: 4-bit ---
    qmm_kernel!(4, 32, f32, f32);
    qmm_kernel!(4, 64, f32, f32);
    qmm_kernel!(4, 128, f32, f32);
    qmm_kernel!(4, 32, f16, f16);
    qmm_kernel!(4, 64, f16, f16);
    qmm_kernel!(4, 128, f16, f16);
    qmm_kernel!(4, 32, bf16, bf16);
    qmm_kernel!(4, 64, bf16, bf16);
    qmm_kernel!(4, 128, bf16, bf16);

    // --- QMM: 8-bit ---
    qmm_kernel!(8, 32, f32, f32);
    qmm_kernel!(8, 64, f32, f32);
    qmm_kernel!(8, 128, f32, f32);
    qmm_kernel!(8, 32, f16, f16);
    qmm_kernel!(8, 64, f16, f16);
    qmm_kernel!(8, 128, f16, f16);
    qmm_kernel!(8, 32, bf16, bf16);
    qmm_kernel!(8, 64, bf16, bf16);
    qmm_kernel!(8, 128, bf16, bf16);
}
