macro_rules! dequant_kernel {
    ($wq:ty, $scalar:ty, $postfix:tt) => {
        paste! {
            pub(crate) fn [< dequantize_ $postfix >](
                wq_packed: *const $wq,
                scale: *const $scalar,
                zero: *const $scalar,
                out: *const $scalar,
                h: i32,
                w: i32
            );
        }
    };
}

pub mod eight_bit {
    use half::{bf16, f16};
    use paste::paste;

    #[allow(dead_code)]
    extern "C" {
        dequant_kernel!(u8, f32, 8bit_u8_kernel_f32);
        dequant_kernel!(u8, f16, 8bit_u8_kernel_f16);
        dequant_kernel!(u8, bf16, 8bit_u8_kernel_bf16);
    }
}

pub mod four_bit {
    use half::{bf16, f16};
    use paste::paste;

    #[allow(dead_code)]
    extern "C" {
        dequant_kernel!(u8, f32, 4bit_u8_kernel_f32);
        dequant_kernel!(u8, f16, 4bit_u8_kernel_f16);
        dequant_kernel!(u8, bf16, 4bit_u8_kernel_bf16);
    }
}

pub mod three_bit {
    use half::{bf16, f16};
    use paste::paste;

    #[allow(dead_code)]
    extern "C" {
        dequant_kernel!(i32, f32, 3bit_32_kernel_f32);
        dequant_kernel!(i32, f16, 3bit_32_kernel_f16);
        dequant_kernel!(i32, bf16, 3bit_32_kernel_bf16);
    }
}

pub mod two_bit {
    use half::{bf16, f16};
    use paste::paste;

    #[allow(dead_code)]
    extern "C" {
        dequant_kernel!(u8, f32, 2bit_u8_kernel_f32);
        dequant_kernel!(u8, f16, 2bit_u8_kernel_f16);
        dequant_kernel!(u8, bf16, 2bit_u8_kernel_bf16);
    }
}

pub mod one_bit {
    use half::{bf16, f16};
    use paste::paste;

    #[allow(dead_code)]
    extern "C" {
        dequant_kernel!(u8, f32, 1bit_u8_kernel_f32);
        dequant_kernel!(u8, f16, 1bit_u8_kernel_f16);
        dequant_kernel!(u8, bf16, 1bit_u8_kernel_bf16);
    }
}
