@echo off
call "E:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
set CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING=1
set CUDA_COMPUTE_CAP=89
set NVCC=%~dp0nvcc_wrapper.bat
cargo build --release -p mistralrs-cli --features cuda %*
