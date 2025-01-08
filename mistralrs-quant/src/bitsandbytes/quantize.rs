#![allow(clippy::excessive_precision)]

use std::f32;

use candle_core::{DType, Device, Result, Tensor};

use super::{BnbDType, BnbLinear, BnbQuantParmas, BnbQuantType};

fn closest_value_index(arr: &[f32], target: f32) -> usize {
    let mut low = 0;
    let mut high = arr.len() - 1;

    // Special case: if there's only one element in the array
    if arr.len() == 1 {
        return 0;
    }

    while low <= high {
        let mid = (low + high) / 2;

        if arr[mid] == target {
            return mid; // Exact match
        } else if arr[mid] < target {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    // After binary search, `low` is the insertion point for the target
    // We need to check which of the two neighbors is closer
    let left_index = if low > 0 { low - 1 } else { 0 };
    let right_index = if low < arr.len() { low } else { arr.len() - 1 };

    let left_diff = (target - arr[left_index]).abs();
    let right_diff = (arr[right_index] - target).abs();

    if left_diff < right_diff {
        left_index
    } else {
        right_index
    }
}

pub fn quantize_block_int8(
    code: &[f32],
    a: &[f32],
    absmax: &mut [f32],
    out: &mut [u8],
    block_end: usize,
    block_idx: usize,
    blocksize: usize,
) {
    // 1. Find absmax in block
    let mut absmax_block = f32::NEG_INFINITY;
    for a_i in a.iter().take(block_end).skip(block_idx) {
        absmax_block = absmax_block.max(a_i.abs());
    }

    absmax[block_idx / blocksize] = absmax_block;

    for i in block_idx..block_end {
        // 2. Divide input value by absmax to normalize into [-1.0, 1.0]
        let normed_value = a[i] / absmax_block;

        // // 3. Perform binary search to find the closest value
        // let mut idx = code
        //     .binary_search_by(|probe| probe.partial_cmp(&normed_value).unwrap())
        //     .unwrap_or_else(|x| {
        //         println!("hello");
        //         x.saturating_sub(1)
        //     });
        let mut idx = closest_value_index(code, normed_value);

        // 4. Check minimal distance
        // Binary search returns the value to the left; check right neighbor if available
        if idx < 255 {
            let dist_left = (normed_value - code[idx]).abs();
            let dist_right = (normed_value - code[idx + 1]).abs();
            if dist_right < dist_left {
                idx += 1;
            }
        }

        // 5. Store index
        out[i] = idx as u8;
    }
}

fn d_quantize_fp4(x: f32) -> u8 {
    // FP4 with bias of 3
    // first bit is a sign
    // subnormals
    // 0b000 = 0
    // 0b001 = 0.0625
    // 0b110 = 2
    // 0b111 = 3
    // 0b100 = 4
    // 0b101 = 6
    // 0b010 = 8
    // 0b011 = 12

    // we do a binary search
    // the pivots are divided by 12 (the FP4 absmax)
    // since we assume input data is in [-1.0, 1.0]

    // Sign bit
    let sign = if x < 0.0 { 0b1000 } else { 0b0000 };
    let x = x.abs();

    if x > 0.29166667 {
        if x > 0.583333 {
            if x > 0.8333333 {
                0b0011 + sign
            } else {
                0b0010 + sign
            }
        } else if x > 0.4166667 {
            0b0101 + sign
        } else {
            0b0100 + sign
        }
    } else if x > 0.0859375 {
        if x > 0.20833333 {
            0b0111 + sign
        } else {
            0b0110 + sign
        }
    } else {
        #[allow(clippy::identity_op)]
        if x > 0.00260417 {
            0b0001 + sign
        } else {
            0b0000 + sign
        }
    }
}

fn d_quantize_nf4(x: f32) -> u8 {
    if x > 0.03979014977812767 {
        if x > 0.3893125355243683 {
            if x > 0.6427869200706482 {
                if x > 0.8614784181118011 {
                    0b1111
                } else {
                    0b1110
                }
            } else if x > 0.5016634166240692 {
                0b1101
            } else {
                0b1100
            }
        } else if x > 0.2035212516784668 {
            if x > 0.2920137718319893 {
                0b1011
            } else {
                0b1010
            }
        } else if x > 0.1202552504837513 {
            0b1001
        } else {
            0b1000
        }
    } else if x > -0.33967943489551544 {
        if x > -0.13791173323988914 {
            if x > -0.045525018125772476 {
                0b0111
            } else {
                0b0110
            }
        } else if x > -0.23460740596055984 {
            0b0101
        } else {
            0b0100
        }
    } else if x > -0.6106329262256622 {
        if x > -0.4599952697753906 {
            0b0011
        } else {
            0b0010
        }
    } else if x > -0.8480964004993439 {
        0b0001
    } else {
        0b0000
    }
}
struct QuantizeResults {
    code: Vec<f32>,
    absmax: Vec<f32>,
    weight: Vec<u8>,
}

const INT8_CODE: [f32; 256] = [
    -0.992968738079071,
    -0.9789062738418579,
    -0.96484375,
    -0.9507812261581421,
    -0.936718761920929,
    -0.922656238079071,
    -0.9085937738418579,
    -0.89453125,
    -0.8804687261581421,
    -0.866406261920929,
    -0.852343738079071,
    -0.8382812738418579,
    -0.82421875,
    -0.8101562261581421,
    -0.796093761920929,
    -0.782031238079071,
    -0.7679687738418579,
    -0.75390625,
    -0.7398437261581421,
    -0.725781261920929,
    -0.7117187976837158,
    -0.6976562738418579,
    -0.68359375,
    -0.6695312261581421,
    -0.655468761920929,
    -0.6414062976837158,
    -0.6273437738418579,
    -0.61328125,
    -0.5992187261581421,
    -0.585156261920929,
    -0.5710937976837158,
    -0.5570312738418579,
    -0.54296875,
    -0.5289062261581421,
    -0.5148437023162842,
    -0.500781238079071,
    -0.48671871423721313,
    -0.47265625,
    -0.4585937261581421,
    -0.44453126192092896,
    -0.43046873807907104,
    -0.4164062440395355,
    -0.40234375,
    -0.3882812261581421,
    -0.37421876192092896,
    -0.36015623807907104,
    -0.3460937440395355,
    -0.33203125,
    -0.3179687261581421,
    -0.30390626192092896,
    -0.28984373807907104,
    -0.2757812738418579,
    -0.26171875,
    -0.24765624105930328,
    -0.23359374701976776,
    -0.21953125298023224,
    -0.20546874403953552,
    -0.19140625,
    -0.17734375596046448,
    -0.16328124701976776,
    -0.14921875298023224,
    -0.13515624403953552,
    -0.12109375,
    -0.10703125596046448,
    -0.09859374910593033,
    -0.09578125923871994,
    -0.09296875447034836,
    -0.09015624970197678,
    -0.08734375238418579,
    -0.08453124761581421,
    -0.08171875774860382,
    -0.07890625298023224,
    -0.07609374821186066,
    -0.07328125089406967,
    -0.07046874612569809,
    -0.0676562562584877,
    -0.06484375149011612,
    -0.062031250447034836,
    -0.05921875312924385,
    -0.05640624836087227,
    -0.053593751043081284,
    -0.05078125,
    -0.047968748956918716,
    -0.04515625163912773,
    -0.04234374687075615,
    -0.039531249552965164,
    -0.03671875223517418,
    -0.033906251192092896,
    -0.031093750149011612,
    -0.028281250968575478,
    -0.025468749925494194,
    -0.02265625074505806,
    -0.019843751564621925,
    -0.017031250521540642,
    -0.014218750409781933,
    -0.011406250298023224,
    -0.009718749672174454,
    -0.009156249463558197,
    -0.008593750186264515,
    -0.008031249977648258,
    -0.0074687497690320015,
    -0.006906250026077032,
    -0.006343749817460775,
    -0.005781250074505806,
    -0.0052187503315508366,
    -0.0046562496572732925,
    -0.004093749914318323,
    -0.0035312497057020664,
    -0.002968749962747097,
    -0.002406249986961484,
    -0.001843750011175871,
    -0.001281249918974936,
    -0.0009437500848434865,
    -0.0008312499849125743,
    -0.0007187500596046448,
    -0.0006062500760890543,
    -0.000493750034365803,
    -0.0003812500217463821,
    -0.0002687500382307917,
    -0.00015625001105945557,
    -8.874999912222847e-05,
    -6.625000241911039e-05,
    -4.374999844003469e-05,
    -2.1249998098937795e-05,
    -7.749999895168003e-06,
    -3.250000190746505e-06,
    -5.500000384017767e-07,
    0.0,
    5.500000384017767e-07,
    3.250000190746505e-06,
    7.749999895168003e-06,
    2.1249998098937795e-05,
    4.374999844003469e-05,
    6.625000241911039e-05,
    8.874999912222847e-05,
    0.00015625001105945557,
    0.0002687500382307917,
    0.0003812500217463821,
    0.000493750034365803,
    0.0006062500760890543,
    0.0007187500596046448,
    0.0008312499849125743,
    0.0009437500848434865,
    0.001281249918974936,
    0.001843750011175871,
    0.002406249986961484,
    0.002968749962747097,
    0.0035312497057020664,
    0.004093749914318323,
    0.0046562496572732925,
    0.0052187503315508366,
    0.005781250074505806,
    0.006343749817460775,
    0.006906250026077032,
    0.0074687497690320015,
    0.008031249977648258,
    0.008593750186264515,
    0.009156249463558197,
    0.009718749672174454,
    0.011406250298023224,
    0.014218750409781933,
    0.017031250521540642,
    0.019843751564621925,
    0.02265625074505806,
    0.025468749925494194,
    0.028281250968575478,
    0.031093750149011612,
    0.033906251192092896,
    0.03671875223517418,
    0.039531249552965164,
    0.04234374687075615,
    0.04515625163912773,
    0.047968748956918716,
    0.05078125,
    0.053593751043081284,
    0.05640624836087227,
    0.05921875312924385,
    0.062031250447034836,
    0.06484375149011612,
    0.0676562562584877,
    0.07046874612569809,
    0.07328125089406967,
    0.07609374821186066,
    0.07890625298023224,
    0.08171875774860382,
    0.08453124761581421,
    0.08734375238418579,
    0.09015624970197678,
    0.09296875447034836,
    0.09578125923871994,
    0.09859374910593033,
    0.10703125596046448,
    0.12109375,
    0.13515624403953552,
    0.14921875298023224,
    0.16328124701976776,
    0.17734375596046448,
    0.19140625,
    0.20546874403953552,
    0.21953125298023224,
    0.23359374701976776,
    0.24765624105930328,
    0.26171875,
    0.2757812738418579,
    0.28984373807907104,
    0.30390626192092896,
    0.3179687261581421,
    0.33203125,
    0.3460937440395355,
    0.36015623807907104,
    0.37421876192092896,
    0.3882812261581421,
    0.40234375,
    0.4164062440395355,
    0.43046873807907104,
    0.44453126192092896,
    0.4585937261581421,
    0.47265625,
    0.48671871423721313,
    0.500781238079071,
    0.5148437023162842,
    0.5289062261581421,
    0.54296875,
    0.5570312738418579,
    0.5710937976837158,
    0.585156261920929,
    0.5992187261581421,
    0.61328125,
    0.6273437738418579,
    0.6414062976837158,
    0.655468761920929,
    0.6695312261581421,
    0.68359375,
    0.6976562738418579,
    0.7117187976837158,
    0.725781261920929,
    0.7398437261581421,
    0.75390625,
    0.7679687738418579,
    0.782031238079071,
    0.796093761920929,
    0.8101562261581421,
    0.82421875,
    0.8382812738418579,
    0.852343738079071,
    0.866406261920929,
    0.8804687261581421,
    0.89453125,
    0.9085937738418579,
    0.922656238079071,
    0.936718761920929,
    0.9507812261581421,
    0.96484375,
    0.9789062738418579,
    0.992968738079071,
    1.0,
];

fn quantize_cpu(a: &[f32], blocksize: usize, quant_ty: BnbQuantType) -> Result<QuantizeResults> {
    // Default params...
    let mut code = if matches!(quant_ty, BnbQuantType::Int8) {
        INT8_CODE.to_vec()
    } else {
        vec![0.; 16]
    };

    let n = a.len();
    let mut absmax = {
        let mut blocks = n / blocksize;
        if n % blocksize > 0 {
            blocks += 1;
        }
        vec![0.; blocks]
    };
    let mut out = if matches!(quant_ty, BnbQuantType::Int8) {
        vec![0; a.len()]
    } else {
        let pack_factor = 2;
        vec![0; (n + 1) / pack_factor]
    };

    // The default code range is adjusted to avoid binary search errors
    code[0] = -1.0;

    match quant_ty {
        BnbQuantType::Int8 => {
            for block_idx in (0..n).step_by(blocksize) {
                let valid_items = if n - block_idx >= blocksize {
                    blocksize
                } else {
                    n - block_idx
                };
                let block_end = block_idx + valid_items;
                quantize_block_int8(
                    &code,
                    a,
                    &mut absmax,
                    &mut out,
                    block_end,
                    block_idx,
                    blocksize,
                );
            }
        }
        BnbQuantType::Nf4 | BnbQuantType::Fp4 => {
            let blocksize = blocksize / 2;
            for block_idx in (0..n / 2).step_by(blocksize) {
                let valid_items = if n - block_idx >= blocksize {
                    blocksize
                } else {
                    n - block_idx
                };
                let block_end = block_idx + valid_items;

                let mut absmax_block = f32::NEG_INFINITY;
                for a_i in a.iter().take(block_end).skip(block_idx) {
                    absmax_block = absmax_block.max(a_i.abs());
                }

                absmax[block_idx / blocksize] = absmax_block;

                absmax_block = 1. / absmax_block;
                for i in block_idx..block_end {
                    let mut packed_4bit = 0u8;
                    match quant_ty {
                        BnbQuantType::Fp4 => {
                            packed_4bit |= d_quantize_fp4(a[2 * i] * absmax_block) << 4;
                            packed_4bit |= d_quantize_fp4(a[2 * i + 1] * absmax_block);
                        }
                        BnbQuantType::Nf4 => {
                            packed_4bit |= d_quantize_nf4(a[2 * i] * absmax_block) << 4;
                            packed_4bit |= d_quantize_nf4(a[2 * i + 1] * absmax_block);
                        }
                        BnbQuantType::Int8 => unreachable!(),
                    }
                    out[i] = packed_4bit;
                }
            }
        }
    }

    Ok(QuantizeResults {
        code,
        absmax,
        weight: out,
    })
}

impl BnbLinear {
    pub fn quantize_onto(
        x: &Tensor,
        quant_ty: BnbQuantType,
        out_ty: BnbDType,
        blocksize: usize,
        device: &Device,
    ) -> Result<Self> {
        let data = x.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let QuantizeResults {
            code,
            absmax,
            weight,
        } = quantize_cpu(&data, blocksize, quant_ty)?;
        let weight_len = weight.len();
        let code_len = code.len();
        let absmax_len = absmax.len();

        let weight = Tensor::from_vec(weight, (weight_len, 1), device)?;
        let code = Tensor::from_vec(code, (code_len,), device)?;
        let absmax = Tensor::from_vec(absmax, (absmax_len,), device)?;

        Ok(Self {
            weight,
            bias: None,
            params: BnbQuantParmas {
                absmax,
                code,
                blocksize,
                shape: Some(x.shape().clone()),
                nested: None,
                offset: None,
                dtype: out_ty,
            },
            quant_ty,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::{bitsandbytes::BnbDType, BnbLinear, BnbQuantType};

    #[test]
    fn test_rountrip_nf4() -> Result<()> {
        let dev = Device::Cpu;

        let data = Tensor::randn(0f32, 1f32, (256, 256), &dev)?;
        let layer = BnbLinear::quantize_onto(&data, BnbQuantType::Nf4, BnbDType::F32, 64, &dev)?;
        let dequant = BnbLinear::dequantize(&layer.weight, &layer.params, BnbQuantType::Nf4)?;

        let err = (data - dequant)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(err < 0.081, "{err}");

        Ok(())
    }

    #[test]
    fn test_rountrip_fp4() -> Result<()> {
        let dev = Device::Cpu;

        let data = Tensor::randn(0f32, 1f32, (256, 256), &dev)?;
        let layer = BnbLinear::quantize_onto(&data, BnbQuantType::Fp4, BnbDType::F32, 64, &dev)?;
        let dequant = BnbLinear::dequantize(&layer.weight, &layer.params, BnbQuantType::Fp4)?;

        let err = (data - dequant)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(err < 0.11, "{err}");

        Ok(())
    }

    #[test]
    fn test_rountrip_int8() -> Result<()> {
        let dev = Device::Cpu;

        let data = Tensor::randn(0f32, 1f32, (256, 256), &dev)?;
        let layer = BnbLinear::quantize_onto(&data, BnbQuantType::Int8, BnbDType::F32, 64, &dev)?;
        let dequant = BnbLinear::dequantize(&layer.weight, &layer.params, BnbQuantType::Int8)?;

        let err = (data - dequant)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(err < 0.01, "{err}");

        Ok(())
    }
}
