#![allow(clippy::excessive_precision)]

use std::fmt::Debug;

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp3, Result, Shape, Tensor, WithDType,
};

use super::{BnbDType, BnbQuantType};

struct DequantizeOp {
    n: usize,
    blocksize: usize,
    shape: Shape,
    quant_ty: BnbQuantType,
    out_ty: BnbDType,
}

fn d_dequantize_nf4(val: u8) -> f32 {
    // the values for this tree were generated by test_normal_map_tree
    // in the file tests/test_functional.py
    if (val & 0b1000) == 0b1000 {
        if (val & 0b0100) == 0b0100 {
            // 1
            if (val & 0b0010) == 0b0010 {
                // 11
                if (val & 0b0001) == 0b0001 {
                    // 111
                    1.0
                } else {
                    0.7229568362236023
                }
            } else if (val & 0b0001) == 0b0001 {
                // 110
                0.5626170039176941
            } else {
                0.44070982933044434
            }
        } else if (val & 0b0010) == 0b0010 {
            // 10
            if (val & 0b0001) == 0b0001 {
                // 101
                0.33791524171829224
            } else {
                0.24611230194568634
            }
        } else if (val & 0b0001) == 0b0001 {
            // 100
            0.16093020141124725
        } else {
            0.07958029955625534
        }
    } else if (val & 0b0100) == 0b0100 {
        // 0
        if (val & 0b0010) == 0b0010 {
            // 01
            if (val & 0b0001) == 0b0001 {
                // 011
                0.0
            } else {
                -0.09105003625154495
            }
        } else if (val & 0b0001) == 0b0001 {
            // 010
            -0.18477343022823334
        } else {
            -0.28444138169288635
        }
    } else if (val & 0b0010) == 0b0010 {
        // 00
        if (val & 0b0001) == 0b0001 {
            // 001
            -0.39491748809814453
        } else {
            -0.5250730514526367
        }
    } else if (val & 0b0001) == 0b0001 {
        // 000
        -0.6961928009986877
    } else {
        -1.0
    }
}

fn d_dequantize_fp4_tree(val: u8, absmax: f32) -> f32 {
    let sign = if (val & 0b1000) == 0b1000 { -1.0 } else { 1.0 };

    if (val & 0b0100) == 0b0100 {
        // 0
        if (val & 0b0010) == 0b0010 {
            // 01
            if (val & 0b0001) == 0b0001 {
                // 111
                0.25000000 * absmax * sign // 1111
            } else {
                0.16666667 * absmax * sign // 1110
            }
        } else if (val & 0b0001) == 0b0001 {
            // 110
            0.50000000 * absmax * sign // 1101
        } else {
            0.33333333 * absmax * sign // 1100
        }
    } else if (val & 0b0010) == 0b0010 {
        // 10
        if (val & 0b0001) == 0b0001 {
            // 101
            1.00000000 * absmax * sign // 1011
        } else {
            0.66666667 * absmax * sign // 1010
        }
    } else if (val & 0b0001) == 0b0001 {
        // 100
        5.208333333e-03 * absmax * sign // 1001
    } else {
        0.00000000 * absmax * sign // 1000
    }
}

impl DequantizeOp {
    fn dequantize_cpu<T: WithDType + Debug>(
        &self,
        input: &[u8],
        absmax: &[f32],
        code: &[f32],
        quant_ty: BnbQuantType,
    ) -> Vec<T> {
        match quant_ty {
            BnbQuantType::Int8 => {
                let mut out = vec![T::zero(); self.n];
                for block_idx in (0..self.n).step_by(self.blocksize) {
                    let valid_items = if self.n - block_idx >= self.blocksize {
                        self.blocksize
                    } else {
                        self.n - block_idx
                    };
                    let block_end = block_idx + valid_items;
                    for i in block_idx..block_end {
                        out[i] = T::from_f64(
                            (code[input[i] as usize] * absmax[block_idx / self.blocksize]) as f64,
                        );
                    }
                }
                out
            }
            BnbQuantType::Fp4 => {
                let mut out = vec![T::zero(); self.shape.elem_count()];
                for block_idx in (0..self.n).step_by(self.blocksize) {
                    let valid_items = if self.n > self.blocksize + block_idx {
                        self.blocksize
                    } else {
                        self.n - block_idx
                    };
                    let block_end = block_idx + valid_items;

                    let local_abs_max = absmax[block_idx / self.blocksize];

                    for i in block_idx..block_end {
                        out[i * 2] =
                            T::from_f64(d_dequantize_fp4_tree(input[i] >> 4, local_abs_max) as f64);
                        out[i * 2 + 1] = T::from_f64(d_dequantize_fp4_tree(
                            input[i] & 0x0F,
                            local_abs_max,
                        ) as f64);
                    }
                }
                out
            }
            BnbQuantType::Nf4 => {
                let mut out = vec![T::zero(); self.shape.elem_count()];
                for block_idx in (0..self.n).step_by(self.blocksize) {
                    let valid_items = if self.n > self.blocksize + block_idx {
                        self.blocksize
                    } else {
                        self.n - block_idx
                    };
                    let block_end = block_idx + valid_items;

                    let local_abs_max = absmax[block_idx / (self.blocksize / 2)];

                    for i in block_idx..block_end {
                        out[i * 2] =
                            T::from_f64((d_dequantize_nf4(input[i] >> 4) * local_abs_max) as f64);
                        out[i * 2 + 1] =
                            T::from_f64((d_dequantize_nf4(input[i] & 0x0F) * local_abs_max) as f64);
                    }
                }
                out
            }
        }
    }
}

impl CustomOp3 for DequantizeOp {
    fn name(&self) -> &'static str {
        "dequantize-bnb"
    }

    fn cpu_fwd(
        &self,
        input_s: &CpuStorage,
        input_l: &candle_core::Layout,
        absmax_s: &CpuStorage,
        absmax_l: &candle_core::Layout,
        code_s: &CpuStorage,
        code_l: &candle_core::Layout,
    ) -> candle_core::Result<(CpuStorage, candle_core::Shape)> {
        if !(input_l.is_contiguous() && absmax_l.is_contiguous() && code_l.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }
        match (input_s, absmax_s, code_s, self.out_ty) {
            (
                CpuStorage::U8(input),
                CpuStorage::F32(absmax),
                CpuStorage::F32(code),
                BnbDType::BF16,
            ) => Ok((
                CpuStorage::BF16(self.dequantize_cpu(input, absmax, code, self.quant_ty)),
                self.shape.clone(),
            )),
            (
                CpuStorage::U8(input),
                CpuStorage::F32(absmax),
                CpuStorage::F32(code),
                BnbDType::F16,
            ) => Ok((
                CpuStorage::F16(self.dequantize_cpu(input, absmax, code, self.quant_ty)),
                self.shape.clone(),
            )),
            (
                CpuStorage::U8(input),
                CpuStorage::F32(absmax),
                CpuStorage::F32(code),
                BnbDType::F32,
            ) => Ok((
                CpuStorage::F32(self.dequantize_cpu(input, absmax, code, self.quant_ty)),
                self.shape.clone(),
            )),
            (i, a, c, t) => candle_core::bail!(
                "Unsupported dtypes for cpu dequant: {:?} input, {:?} absmax, {:?} code, {:?} out",
                i.dtype(),
                a.dtype(),
                c.dtype(),
                t
            ),
        }
    }
}

pub fn dequantize(
    input: &Tensor,
    absmax: &Tensor,
    code: &Tensor,
    shape: Shape,
    blocksize: usize,
    quant_ty: BnbQuantType,
    out_ty: BnbDType,
) -> Result<Tensor> {
    input.apply_op3(
        absmax,
        code,
        DequantizeOp {
            n: input.elem_count(),
            blocksize,
            shape,
            quant_ty,
            out_ty,
        },
    )
}
