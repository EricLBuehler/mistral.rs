use candle_core::{
    cuda::cudarc::{
        driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
    },
    Result,
};

const DELAY_CYCLES: u64 = 50_000_000;
const DELAY_PTX: &str = r#"
.version 6.0
.target sm_70
.address_size 64
.visible .entry delay_cycles(.param .u64 cycles) {
    .reg .pred p;
    .reg .b64 start, now, elapsed, duration;
    ld.param.u64 duration, [cycles];
    mov.u64 start, %clock64;
delay:
    mov.u64 now, %clock64;
    sub.u64 elapsed, now, start;
    setp.lt.u64 p, elapsed, duration;
    @p bra delay;
    ret;
}
"#;

pub(crate) fn delay_stream(
    stream: &std::sync::Arc<CudaStream>,
) -> Result<std::sync::Arc<CudaModule>> {
    let module = stream
        .context()
        .load_module(Ptx::from_src(DELAY_PTX))
        .map_err(candle_core::Error::wrap)?;
    let function = module
        .load_function("delay_cycles")
        .map_err(candle_core::Error::wrap)?;
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&DELAY_CYCLES)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(candle_core::Error::wrap)?;
    Ok(module)
}
