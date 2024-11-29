use candle_core::{DType, Device, Result, Tensor, D};

#[derive(Debug, Clone)]
pub struct ImatrixLayerStats {
    pub row_counts: usize,
    pub ncalls: usize,
    pub row_accum: Tensor,
}

impl ImatrixLayerStats {
    pub fn new(w: &Tensor, device: &Device) -> Result<Self> {
        Ok(Self {
            row_counts: 0,
            ncalls: 0,
            row_accum: Tensor::zeros((w.dim(1)?,), DType::F32, device)?,
        })
    }

    pub fn process(&mut self, inp: &Tensor) -> Result<()> {
        let inp = inp.reshape(((), inp.dim(D::Minus1)?))?;
        self.ncalls += 1;
        self.row_counts += inp.dim(D::Minus1)?;
        self.row_accum = (&self.row_accum + inp.to_dtype(DType::F32)?.sqr()?.sum(0)?)?;
        Ok(())
    }

    pub fn compute_imatrix(&self) -> Result<Tensor> {
        (&self.row_accum / self.row_counts as f64)? * self.ncalls as f64
    }
}
// let mut row_counts = 0f64;
// let mut ncall = 0f64;
// let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

// for _ in 0..10 {
//     let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
//     let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
//     let res = lhs.matmul(&rhs)?;

//     // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
//     values = (values + res.sqr()?.sum(0)?)?;
//     row_counts += res.dim(0)? as f64;
//     ncall += 1.;
// }

// // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
// let out = ((values / row_counts)? * ncall)?;
// let imatrix = out.to_vec1::<f32>()?;
