#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

#[cfg(feature = "metal")]
use std::sync::atomic::AtomicUsize;

use crate::MemoryUsage;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::MatMul;

use crate::attention::SdpaParams;

use super::cpu;

#[cfg(feature = "metal")]
/// Initial, sentinel value is usize::MAX
static METAL_VERSION_CACHE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn supports_attn_softmax() -> Result<bool> {
    #[cfg(feature = "metal")]
    {
        use std::sync::atomic::Ordering;
        let cache = METAL_VERSION_CACHE.load(Ordering::Relaxed);

        let version = if cache != usize::MAX {
            cache
        } else {
            // echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P -

            use std::process::{Command, Stdio};

            // Create the `echo` command and pipe its output into `xcrun`
            let mut echo = Command::new("echo")
                .arg("__METAL_VERSION__")
                .stdout(Stdio::piped())
                .spawn()
                .expect("Failed to start echo command");

            echo.wait()?;

            // Run the `xcrun` command, taking input from the `echo` command's output
            let output = Command::new("xcrun")
                .arg("-sdk")
                .arg("macosx")
                .arg("metal")
                .arg("-E")
                .arg("-x")
                .arg("metal")
                .arg("-P")
                .arg("-")
                .stdin(echo.stdout.unwrap())
                .output()
                .expect("Failed to run xcrun command");

            // Handle the output
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout)
                    .split('\n')
                    .nth(1)
                    .unwrap()
                    .trim()
                    .to_string()
                    .parse::<usize>()
                    .unwrap();
                METAL_VERSION_CACHE.store(version, Ordering::Relaxed);
                version
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("Error:\n{}", stderr);
            }
        };
        // Attn softmax is only supported for metal >= 310
        Ok(version >= 310)
    }

    #[cfg(not(feature = "metal"))]
    Ok(true)
}

/// Not *really* sure why this is necessary but it is.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    // If less that 4 GB available, synchronize
    if MemoryUsage.get_memory_available(device)? < 4 * 1024 * (1024 * 1024) {
        device.synchronize()?;
    }
    Ok(())
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    if q.device().is_cpu() {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        match q.dtype() {
            DType::F32 => cpu::run_flash_attn_cpu::<f32>(&q, &k, &v, mask, sdpa_params),
            DType::F16 => cpu::run_flash_attn_cpu::<half::f16>(&q, &k, &v, mask, sdpa_params),
            DType::BF16 => cpu::run_flash_attn_cpu::<half::bf16>(&q, &k, &v, mask, sdpa_params),
            _ => Err(candle_core::Error::Msg("Unsupported data type".into())),
        }
    } else {
        maybe_synchronize(q.device())?;

        // Use faster softmax if mask is rank 2 or it's rank 3
        if mask.is_some_and(|mask| mask.rank() == 2 || mask.rank() == 3) && supports_attn_softmax()?
        {
            let mask = match mask {
                Some(mask) if mask.rank() == 3 || mask.rank() == 2 => mask.clone(),
                _ => candle_core::bail!("unsupported mask {mask:?}"),
            };

            let mut att = MatMul.matmul(q, &k.t()?)?;

            candle_nn::ops::inplace_attn_softmax_last_dim(
                &mut att,
                &mask.contiguous()?,
                sdpa_params.softmax_scale / sdpa_params.softcap.unwrap_or(1.0),
            )?;

            if let Some(softcap) = sdpa_params.softcap {
                att = (att.tanh()? * softcap as f64)?;
            }

            MatMul.matmul(&att, v)
        } else if let Some(mask) = mask {
            let mut att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                att = (att / softcap as f64)?;
                att = att.tanh()?;
                att = (att * softcap as f64)?;
            }

            att = att.broadcast_add(mask)?;
            candle_nn::ops::inplace_softmax_last_dim(&mut att)?;

            MatMul.matmul(&att, v)
        } else {
            let mut att = MatMul.matmul_affine_mul(q, &k.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                att = (att / softcap as f64)?;
                att = att.tanh()?;
                att = (att * softcap as f64)?;
            }

            candle_nn::ops::inplace_softmax_last_dim(&mut att)?;
            MatMul.matmul(&att, v)
        }
    }
}
