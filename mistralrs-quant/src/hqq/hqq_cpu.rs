use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, WithDType};

/*
 8 bit
*/
pub(crate) struct Dequant8Bit {
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Dequant8Bit {
    fn dequantize<T: WithDType>(&self, w: &[u8], s: &[T], z: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(w.len());
        for (i, w) in w.iter().enumerate() {
            let j = i % self.w;
            out[i] = (T::from_f64(*w as f64) - z[j]) * s[j];
        }
        out
    }
}

impl CustomOp3 for Dequant8Bit {
    fn name(&self) -> &'static str {
        "dequant-hqq-8bit"
    }
    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        l_w: &Layout,
        s: &CpuStorage,
        l_s: &Layout,
        z: &CpuStorage,
        l_z: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::U8(w_slice) = w else {
            candle_core::bail!("Weight must be u8, HQQ dequant 8-bit");
        };
        if !(l_w.is_contiguous() && l_s.is_contiguous() && l_z.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }
        match (s, z) {
            (CpuStorage::F32(s_slice), CpuStorage::F32(z_slice)) => Ok((
                CpuStorage::F32(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[self.h, self.w]),
            )),
            (CpuStorage::F16(s_slice), CpuStorage::F16(z_slice)) => Ok((
                CpuStorage::F16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[self.h, self.w]),
            )),
            (CpuStorage::BF16(s_slice), CpuStorage::BF16(z_slice)) => Ok((
                CpuStorage::BF16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[self.h, self.w]),
            )),
            (_, _) => candle_core::bail!("Dtype mismatch, expected one of f32, f16, bf16"),
        }
    }
}

/*
 4 bit
*/
pub(crate) struct Dequant4Bit {
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Dequant4Bit {
    fn dequantize<T: WithDType>(&self, w: &[u8], s: &[T], z: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(w.len());
        for (i, w) in w.iter().enumerate() {
            let j = i % self.w;
            let nrows = self.h * self.w;
            out[i] = (T::from_f64(((*w & 0xF0) >> 4) as f64) - z[j]) * s[j];
            out[i + nrows] = (T::from_f64((*w & 0x0F) as f64) - z[j]) * s[j];
        }
        out
    }
}

impl CustomOp3 for Dequant4Bit {
    fn name(&self) -> &'static str {
        "dequant-hqq-4bit"
    }
    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        l_w: &Layout,
        s: &CpuStorage,
        l_s: &Layout,
        z: &CpuStorage,
        l_z: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        const PACK_FACTOR: usize = 2;

        let CpuStorage::U8(w_slice) = w else {
            candle_core::bail!("Weight must be u8, HQQ dequant 4-bit");
        };
        if !(l_w.is_contiguous() && l_s.is_contiguous() && l_z.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }
        match (s, z) {
            (CpuStorage::F32(s_slice), CpuStorage::F32(z_slice)) => Ok((
                CpuStorage::F32(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::F16(s_slice), CpuStorage::F16(z_slice)) => Ok((
                CpuStorage::F16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::BF16(s_slice), CpuStorage::BF16(z_slice)) => Ok((
                CpuStorage::BF16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (_, _) => candle_core::bail!("Dtype mismatch, expected one of f32, f16, bf16"),
        }
    }
}

/*
 2 bit
*/
pub(crate) struct Dequant2Bit {
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Dequant2Bit {
    fn dequantize<T: WithDType>(&self, w: &[u8], s: &[T], z: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(w.len());
        for (i, w) in w.iter().enumerate() {
            let j = i % self.w;
            let nrows = self.h * self.w;
            out[i] = (T::from_f64(((*w & 0xC0) >> 6) as f64) - z[j]) * s[j];
            out[i + nrows] = (T::from_f64(((*w & 0x30) >> 4) as f64) - z[j]) * s[j];
            out[i + nrows * 2] = (T::from_f64(((*w & 0x0C) >> 2) as f64) - z[j]) * s[j];
            out[i + nrows * 3] = (T::from_f64((*w & 0x03) as f64) - z[j]) * s[j];
        }
        out
    }
}

impl CustomOp3 for Dequant2Bit {
    fn name(&self) -> &'static str {
        "dequant-hqq-2bit"
    }
    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        l_w: &Layout,
        s: &CpuStorage,
        l_s: &Layout,
        z: &CpuStorage,
        l_z: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        const PACK_FACTOR: usize = 4;

        let CpuStorage::U8(w_slice) = w else {
            candle_core::bail!("Weight must be u8, HQQ dequant 2-bit");
        };
        if !(l_w.is_contiguous() && l_s.is_contiguous() && l_z.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }
        match (s, z) {
            (CpuStorage::F32(s_slice), CpuStorage::F32(z_slice)) => Ok((
                CpuStorage::F32(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::F16(s_slice), CpuStorage::F16(z_slice)) => Ok((
                CpuStorage::F16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::BF16(s_slice), CpuStorage::BF16(z_slice)) => Ok((
                CpuStorage::BF16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (_, _) => candle_core::bail!("Dtype mismatch, expected one of f32, f16, bf16"),
        }
    }
}

/*
 1 bit
*/
pub(crate) struct Dequant1Bit {
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Dequant1Bit {
    fn dequantize<T: WithDType>(&self, w: &[u8], s: &[T], z: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(w.len());
        for (i, w) in w.iter().enumerate() {
            let j = i % self.w;
            let nrows = self.h * self.w;
            out[i] = (T::from_f64(((*w & 0x80) >> 7) as f64) - z[j]) * s[j];
            out[i + nrows] = (T::from_f64(((*w & 0x40) >> 6) as f64) - z[j]) * s[j];
            out[i + nrows * 2] = (T::from_f64(((*w & 0x20) >> 5) as f64) - z[j]) * s[j];
            out[i + nrows * 3] = (T::from_f64(((*w & 0x10) >> 4) as f64) - z[j]) * s[j];
            out[i + nrows * 4] = (T::from_f64(((*w & 0x08) >> 3) as f64) - z[j]) * s[j];
            out[i + nrows * 5] = (T::from_f64(((*w & 0x04) >> 2) as f64) - z[j]) * s[j];
            out[i + nrows * 6] = (T::from_f64(((*w & 0x02) >> 1) as f64) - z[j]) * s[j];
            out[i + nrows * 7] = (T::from_f64((*w & 0x01) as f64) - z[j]) * s[j];
        }
        out
    }
}

impl CustomOp3 for Dequant1Bit {
    fn name(&self) -> &'static str {
        "dequant-hqq-1bit"
    }
    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        l_w: &Layout,
        s: &CpuStorage,
        l_s: &Layout,
        z: &CpuStorage,
        l_z: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        const PACK_FACTOR: usize = 8;

        let CpuStorage::U8(w_slice) = w else {
            candle_core::bail!("Weight must be u8, HQQ dequant 1-bit");
        };
        if !(l_w.is_contiguous() && l_s.is_contiguous() && l_z.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }
        match (s, z) {
            (CpuStorage::F32(s_slice), CpuStorage::F32(z_slice)) => Ok((
                CpuStorage::F32(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::F16(s_slice), CpuStorage::F16(z_slice)) => Ok((
                CpuStorage::F16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::BF16(s_slice), CpuStorage::BF16(z_slice)) => Ok((
                CpuStorage::BF16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (_, _) => candle_core::bail!("Dtype mismatch, expected one of f32, f16, bf16"),
        }
    }
}

/*
 3 bit
*/
pub(crate) struct Dequant3Bit {
    pub(crate) h: usize,
    pub(crate) w: usize,
}

impl Dequant3Bit {
    fn dequantize<T: WithDType>(&self, w: &[i32], s: &[T], z: &[T]) -> Vec<T> {
        let mut out = Vec::with_capacity(w.len());
        for (i, w) in w.iter().enumerate() {
            let j = i % self.w;
            let nrows = self.h * self.w;
            out[i] = (T::from_f64(((*w & 0x38000000) >> 27) as f64) - z[j]) * s[j];
            out[i + nrows] = (T::from_f64(((*w & 0x07000000) >> 24) as f64) - z[j]) * s[j];
            out[i + nrows * 2] = (T::from_f64(((*w & 0x00E00000) >> 21) as f64) - z[j]) * s[j];
            out[i + nrows * 3] = (T::from_f64(((*w & 0x001C0000) >> 18) as f64) - z[j]) * s[j];
            out[i + nrows * 4] = (T::from_f64(((*w & 0x00038000) >> 15) as f64) - z[j]) * s[j];
            out[i + nrows * 5] = (T::from_f64(((*w & 0x00007000) >> 12) as f64) - z[j]) * s[j];
            out[i + nrows * 6] = (T::from_f64(((*w & 0x00000E00) >> 9) as f64) - z[j]) * s[j];
            out[i + nrows * 7] = (T::from_f64(((*w & 0x000001C0) >> 6) as f64) - z[j]) * s[j];
            out[i + nrows * 8] = (T::from_f64(((*w & 0x00000038) >> 3) as f64) - z[j]) * s[j];
            out[i + nrows * 9] = (T::from_f64((*w & 0x00000007) as f64) - z[j]) * s[j];
        }
        out
    }
}

impl CustomOp3 for Dequant3Bit {
    fn name(&self) -> &'static str {
        "dequant-hqq-3bit"
    }
    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        l_w: &Layout,
        s: &CpuStorage,
        l_s: &Layout,
        z: &CpuStorage,
        l_z: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        const PACK_FACTOR: usize = 10;

        let CpuStorage::I32(w_slice) = w else {
            candle_core::bail!("Weight must be i32, HQQ dequant 3-bit");
        };
        if !(l_w.is_contiguous() && l_s.is_contiguous() && l_z.is_contiguous()) {
            candle_core::bail!("All inputs must be contiguous");
        }
        match (s, z) {
            (CpuStorage::F32(s_slice), CpuStorage::F32(z_slice)) => Ok((
                CpuStorage::F32(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::F16(s_slice), CpuStorage::F16(z_slice)) => Ok((
                CpuStorage::F16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (CpuStorage::BF16(s_slice), CpuStorage::BF16(z_slice)) => Ok((
                CpuStorage::BF16(self.dequantize(w_slice, s_slice, z_slice)),
                Shape::from_dims(&[PACK_FACTOR * self.h, self.w]),
            )),
            (_, _) => candle_core::bail!("Dtype mismatch, expected one of f32, f16, bf16"),
        }
    }
}
