use candle_core::{cuda_backend::kernels, Device, Result};

const PTX_ENTRY_PREFIX: &str = ".visible .entry ";

pub(crate) fn preload_candle_ptx(device: &Device) -> Result<usize> {
    let Device::Cuda(cuda_device) = device else {
        return Ok(0);
    };

    let modules = [
        &kernels::AFFINE,
        &kernels::BINARY,
        &kernels::CAST,
        &kernels::CONV,
        &kernels::FILL,
        &kernels::INDEXING,
        &kernels::QUANTIZED,
        &kernels::REDUCE,
        &kernels::SORT,
        &kernels::TERNARY,
        &kernels::UNARY,
    ];

    let mut count = 0;
    for module in modules {
        for entry in ptx_entry_names(module.ptx()) {
            let _func = cuda_device.get_or_load_func(entry, module)?;
            count += 1;
        }
    }

    Ok(count)
}

fn ptx_entry_names(ptx: &str) -> impl Iterator<Item = &str> {
    ptx.lines().filter_map(|line| {
        let entry = line.trim_start().strip_prefix(PTX_ENTRY_PREFIX)?;
        let (name, _) = entry.split_once('(')?;
        let name = name.trim();
        (!name.is_empty()).then_some(name)
    })
}
