//! FFI bindings for the cuBLASLt library, focusing on matrix multiplication (GEMM).
//!
//! These bindings allow Rust code to call cuBLASLt functions for high-performance
//! GEMM operations, including support for FP8 data types.
//!
//! Most handle types are opaque pointers. Enums mirror the definitions in
//! `cublas_v2.h` and `cublasLt.h`. It's crucial that the enum variants and their
//! raw integer values match the NVIDIA headers precisely.
//!
//! Note: These definitions may overlap with or should eventually be reconciled with
//! FFI bindings potentially available in `candle-core` or its dependencies like `cudarc`.
//! This module provides a self-contained set for the specific FP8 GEMM needs here.

#[cfg(feature = "cuda")]
mod cuda_bindings {
    use candle_core::cuda::ffi; // For existing ffi::CUdeviceptr, ffi::cudaStream_t
    use std::ffi::c_void;

    // Opaque handles
    /// Opaque handle to the cuBLASLt library context. Created by `cublasLtCreate`.
    #[repr(C)] #[allow(non_camel_case_types)] pub struct cublasLtContext { _private: [u8; 0] }
    #[allow(non_camel_case_types)] pub type cublasLtHandle_t = *mut cublasLtContext;

    /// Opaque handle to a matrix multiplication descriptor. Created by `cublasLtMatmulDescCreate`.
    #[repr(C)] #[allow(non_camel_case_types)] pub struct cublasLtMatmulDesc { _private: [u8; 0] }
    #[allow(non_camel_case_types)] pub type cublasLtMatmulDesc_t = *mut cublasLtMatmulDesc;

    /// Opaque handle to a matrix layout descriptor. Created by `cublasLtMatrixLayoutCreate`.
    #[repr(C)] #[allow(non_camel_case_types)] pub struct cublasLtMatrixLayout { _private: [u8; 0] }
    #[allow(non_camel_case_types)] pub type cublasLtMatrixLayout_t = *mut cublasLtMatrixLayout;

    /// Opaque handle for heuristic search preferences. Created by `cublasLtMatmulPreferenceCreate`.
    #[repr(C)] #[allow(non_camel_case_types)] pub struct cublasLtMatmulPreference { _private: [u8; 0] }
    #[allow(non_camel_case_types)] pub type cublasLtMatmulPreference_t = *mut cublasLtMatmulPreference;

    /// Opaque structure describing a matrix multiplication algorithm.
    /// This can be obtained from heuristics and passed to `cublasLtMatmul`.
    #[repr(C)] #[derive(Debug, Clone, Copy)] #[allow(non_camel_case_types)] pub struct cublasLtMatmulAlgo_t { pub data: [u64; 8], }

    /// cuBLAS status type, typically an integer. `0` usually indicates success (`CUBLAS_STATUS_SUCCESS`).
    #[allow(non_camel_case_types)] pub type cublasStatus_t = i32;

    /// CUDA data types enum, mirroring `library_types.h` and `cuda_fp8.h`.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cudaDataType_t {
        CUDA_R_32F = 0,
        CUDA_R_16F = 2,
        CUDA_R_16BF = 14,
        CUDA_R_8I = 3,
        CUDA_R_8F_E4M3 = 28,
        CUDA_R_8F_E5M2 = 29,
    }

    /// cublasOperation_t from `cublas_api.h`.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cublasOperation_t {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2,
    }

    /// cublasComputeType_t from `cublas_api.h`.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cublasComputeType_t {
        CUBLAS_COMPUTE_16F = 64,
        CUBLAS_COMPUTE_32F = 68,
        CUBLAS_COMPUTE_32F_FAST_TF32 = 74,
    }

    /// Attributes for `cublasLtMatmulDesc_t`.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cublasLtMatmulDescAttributes_t {
        CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
        CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
        CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,
        CUBLASLT_MATMUL_DESC_TRANSA = 3,
        CUBLASLT_MATMUL_DESC_TRANSB = 4,
        CUBLASLT_MATMUL_DESC_TRANSC = 5,
        CUBLASLT_MATMUL_DESC_FILL_MODE = 6,
        CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
        CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 9,
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 10,
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 11,
        CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 12,
        CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 13,
        CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = 14,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 15,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 16,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 17,
        CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 22,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 32,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 33,
        CUBLASLT_MATMUL_DESC_C_SCALE_MODE = 34,
        CUBLASLT_MATMUL_DESC_D_SCALE_MODE = 35,
        CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER = 37,
    }

    /// Attributes for `cublasLtMatrixLayout_t`.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cublasLtMatrixLayoutAttribute_t {
        CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
        CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
        CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
        CUBLASLT_MATRIX_LAYOUT_COLS = 3,
        CUBLASLT_MATRIX_LAYOUT_LD = 4,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    }

    /// Epilogue operations for `cublasLtMatmul`.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cublasLtEpilogue_t {
        CUBLASLT_EPILOGUE_DEFAULT = 1,
        CUBLASLT_EPILOGUE_BIAS = 4,
    }

    /// Defines how matrix scale factors are interpreted.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[allow(non_camel_case_types)]
    pub enum cublasLtMatmulMatrixScale_t {
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F = 0,
        CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F = 3,
    }

    /// Result from `cublasLtMatmulAlgoGetHeuristic`.
    #[repr(C)]
    #[derive(Debug)]
    #[allow(non_camel_case_types)]
    pub struct cublasLtMatmulHeuristicResult_t {
        pub algo: cublasLtMatmulAlgo_t,
        pub workspaceSize: usize,
        pub state: cublasStatus_t,
        pub wavesCount: f32,
        pub reserved: [i32; 4],
    }

    #[link(name = "cublas")]
    extern "C" {
        pub fn cublasLtCreate(lightHandle: *mut cublasLtHandle_t) -> cublasStatus_t;
        pub fn cublasLtDestroy(lightHandle: cublasLtHandle_t) -> cublasStatus_t;

        pub fn cublasLtMatmulDescCreate(
            matmulDesc: *mut cublasLtMatmulDesc_t,
            computeType: cublasComputeType_t,
            scaleType: cudaDataType_t,
        ) -> cublasStatus_t;
        pub fn cublasLtMatmulDescDestroy(matmulDesc: cublasLtMatmulDesc_t) -> cublasStatus_t;
        pub fn cublasLtMatmulDescSetAttribute(
            matmulDesc: cublasLtMatmulDesc_t,
            attr: cublasLtMatmulDescAttributes_t,
            buf: *const c_void,
            sizeInBytes: usize,
        ) -> cublasStatus_t;

        pub fn cublasLtMatrixLayoutCreate(
            matLayout: *mut cublasLtMatrixLayout_t,
            type_: cudaDataType_t,
            rows: u64,
            cols: u64,
            ld: i64,
        ) -> cublasStatus_t;
        pub fn cublasLtMatrixLayoutDestroy(matLayout: cublasLtMatrixLayout_t) -> cublasStatus_t;
        pub fn cublasLtMatrixLayoutSetAttribute(
            matLayout: cublasLtMatrixLayout_t,
            attr: cublasLtMatrixLayoutAttribute_t,
            buf: *const c_void,
            sizeInBytes: usize,
        ) -> cublasStatus_t;

        pub fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> cublasStatus_t;
        pub fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> cublasStatus_t;

        pub fn cublasLtMatmulAlgoGetHeuristic(
            lightHandle: cublasLtHandle_t,
            operationDesc: cublasLtMatmulDesc_t,
            Adesc: cublasLtMatrixLayout_t,
            Bdesc: cublasLtMatrixLayout_t,
            Cdesc: cublasLtMatrixLayout_t,
            Ddesc: cublasLtMatrixLayout_t,
            preference: cublasLtMatmulPreference_t,
            requestedAlgoCount: i32,
            heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
            returnAlgoCount: *mut i32,
        ) -> cublasStatus_t;

        pub fn cublasLtMatmul(
            lightHandle: cublasLtHandle_t,
            computeDesc: cublasLtMatmulDesc_t,
            alpha: *const c_void,
            A: ffi::CUdeviceptr,
            Adesc: cublasLtMatrixLayout_t,
            B: ffi::CUdeviceptr,
            Bdesc: cublasLtMatrixLayout_t,
            beta: *const c_void,
            C: ffi::CUdeviceptr,
            Cdesc: cublasLtMatrixLayout_t,
            D: ffi::CUdeviceptr,
            Ddesc: cublasLtMatrixLayout_t,
            algo: *const cublasLtMatmulAlgo_t,
            workspace: ffi::CUdeviceptr,
            workspaceSizeInBytes: usize,
            stream: ffi::cudaStream_t,
        ) -> cublasStatus_t;
    }
}

#[cfg(feature = "cuda")]
pub use cuda_bindings::*;
