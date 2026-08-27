use std::time::Duration;

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use rand_isaac::Isaac64Rng;

#[cfg(feature = "cuda")]
use crate::{prefix_cacher::PrefixCacheManagerV2, sequence::Sequence};

#[cfg(feature = "cuda")]
use super::{
    sampling::{self, CudaTokenBatchSubmission},
    ForwardInputsResult, ForwardStepResult, Pipeline,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepLookahead {
    #[default]
    Disabled,
    OneToken,
}

impl StepLookahead {
    pub(crate) fn is_enabled(self) -> bool {
        matches!(self, Self::OneToken)
    }
}

pub(crate) struct StepCompletion {
    duration: Duration,
    #[cfg(feature = "cuda")]
    cuda_tail: Option<CudaDecodeTail>,
}

impl StepCompletion {
    pub(crate) fn ready(duration: Duration) -> Self {
        Self {
            duration,
            #[cfg(feature = "cuda")]
            cuda_tail: None,
        }
    }

    pub(crate) fn duration(&self) -> Duration {
        self.duration
    }

    #[cfg(feature = "cuda")]
    fn with_cuda_tail(duration: Duration, cuda_tail: Option<CudaDecodeTail>) -> Self {
        Self {
            duration,
            cuda_tail,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn into_cuda_tail(mut self) -> Option<CudaDecodeTail> {
        self.cuda_tail.take()
    }
}

pub struct StepSubmission {
    inner: StepSubmissionKind,
}

pub(crate) enum StepSubmissionKind {
    Ready(StepCompletion),
    #[cfg(feature = "cuda")]
    Cuda(CudaStepSubmission),
}

impl StepSubmission {
    pub(crate) fn ready(duration: Duration) -> Self {
        Self {
            inner: StepSubmissionKind::Ready(StepCompletion::ready(duration)),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(submission: CudaStepSubmission) -> Self {
        Self {
            inner: StepSubmissionKind::Cuda(submission),
        }
    }

    pub(crate) fn into_inner(self) -> StepSubmissionKind {
        self.inner
    }

    pub(crate) fn into_ready(self) -> Option<StepCompletion> {
        match self.inner {
            StepSubmissionKind::Ready(completion) => Some(completion),
            #[cfg(feature = "cuda")]
            StepSubmissionKind::Cuda(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_has_tail(&self) -> bool {
        matches!(&self.inner, StepSubmissionKind::Cuda(step) if step.has_tail())
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaDecodeTail {
    result: Option<ForwardStepResult>,
    stream: Arc<CudaStream>,
    pending: bool,
}

#[cfg(feature = "cuda")]
impl CudaDecodeTail {
    fn new(result: ForwardStepResult, stream: Arc<CudaStream>) -> Self {
        Self {
            result: Some(result),
            stream,
            pending: true,
        }
    }

    fn causal_logits(&self) -> candle_core::Result<&candle_core::Tensor> {
        let result = self.result.as_ref().expect("CUDA decode tail was consumed");
        match &result.output {
            ForwardInputsResult::CausalGeneration { logits } => Ok(logits),
            _ => candle_core::bail!("CUDA decode tail does not contain causal logits"),
        }
    }

    fn launch(&self) -> Option<&super::CudaDecodeGraphLaunch> {
        self.result
            .as_ref()
            .expect("CUDA decode tail was consumed")
            .cuda_decode
            .as_ref()
    }

    fn take_result(&mut self) -> ForwardStepResult {
        self.pending = false;
        self.result.take().expect("CUDA decode tail was consumed")
    }

    pub(crate) fn batch_size(&self) -> candle_core::Result<usize> {
        self.causal_logits()?.dim(0)
    }

    pub(crate) fn synchronize(&mut self) -> candle_core::Result<()> {
        if self.pending {
            self.stream
                .synchronize()
                .map_err(candle_core::Error::wrap)?;
            self.pending = false;
        }
        Ok(())
    }

    pub(crate) fn drain(mut self) -> candle_core::Result<()> {
        self.synchronize()
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaDecodeTail {
    fn drop(&mut self) {
        if self.pending {
            let _ = self.stream.synchronize();
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaStepSubmission {
    current: Option<CudaTokenBatchSubmission>,
    tail: Option<CudaDecodeTail>,
    duration: Duration,
}

#[cfg(feature = "cuda")]
impl CudaStepSubmission {
    fn new(
        current: CudaTokenBatchSubmission,
        tail: Option<CudaDecodeTail>,
        duration: Duration,
    ) -> Self {
        Self {
            current: Some(current),
            tail,
            duration,
        }
    }

    pub(crate) fn has_tail(&self) -> bool {
        self.tail.is_some()
    }

    pub(crate) fn into_parts(mut self) -> (CudaTokenBatchSubmission, CudaStepPending) {
        let current = self
            .current
            .take()
            .expect("CUDA step submission was completed");
        let pending = CudaStepPending {
            tail: self.tail.take(),
            duration: self.duration,
            batch_size: current.batch_size(),
        };
        (current, pending)
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaStepPending {
    tail: Option<CudaDecodeTail>,
    duration: Duration,
    batch_size: usize,
}

#[cfg(feature = "cuda")]
impl CudaStepPending {
    pub(crate) fn finish(mut self, token_ids: Vec<u32>) -> candle_core::Result<CudaStepCompletion> {
        if token_ids.len() != self.batch_size {
            candle_core::bail!(
                "CUDA step completion has {} tokens for a batch of {}",
                token_ids.len(),
                self.batch_size
            );
        }
        Ok(CudaStepCompletion {
            token_ids,
            tail: self.tail.take(),
            duration: self.duration,
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaStepCompletion {
    token_ids: Vec<u32>,
    tail: Option<CudaDecodeTail>,
    duration: Duration,
}

#[cfg(feature = "cuda")]
impl CudaStepCompletion {
    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    #[cfg(test)]
    pub(crate) fn has_tail(&self) -> bool {
        self.tail.is_some()
    }

    pub(crate) fn synchronize_tail(&mut self) -> candle_core::Result<()> {
        if let Some(tail) = self.tail.as_mut() {
            tail.synchronize()?;
        }
        Ok(())
    }

    pub(crate) async fn finish(
        mut self,
        pipeline: &dyn Pipeline,
        seqs: &mut [&mut Sequence],
        commit_rows: &[bool],
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
    ) -> candle_core::Result<StepCompletion> {
        sampling::finish_cuda_token_batch(
            pipeline,
            seqs,
            self.token_ids,
            commit_rows,
            prefix_cacher,
            disable_eos_stop,
        )
        .await?;
        Ok(StepCompletion::with_cuda_tail(
            self.duration,
            self.tail.take(),
        ))
    }
}

#[cfg(feature = "cuda")]
pub(crate) enum CudaTailSubmission {
    Submitted(CudaStepSubmission),
    Unsupported(CudaDecodeTail),
}

#[cfg(feature = "cuda")]
pub(crate) fn submit_forward_lookahead<P: Pipeline + ?Sized>(
    pipeline: &mut P,
    seqs: &[&mut Sequence],
    result: ForwardStepResult,
    duration: Duration,
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
) -> candle_core::Result<Result<CudaStepSubmission, ForwardStepResult>> {
    let started = Instant::now();
    let ForwardInputsResult::CausalGeneration { logits } = &result.output else {
        return Ok(Err(result));
    };
    let Some(launch) = result.cuda_decode.as_ref() else {
        return Ok(Err(result));
    };
    if launch.real_batch() != seqs.len()
        || !sampling::can_launch_one_token_lookahead(seqs, pipeline.get_metadata().max_seq_len)
    {
        return Ok(Err(result));
    }
    let stream = launch.graph_stream().clone();
    let Some(current) =
        sampling::try_submit_cuda_token_batch(logits, seqs, launch.resident_input(), rng)?
    else {
        return Ok(Err(result));
    };
    current.wait_on(&stream)?;

    let ForwardStepResult {
        output: _,
        cuda_decode: Some(launch),
    } = result
    else {
        unreachable!("CUDA lookahead launch disappeared after submission")
    };
    let tail = pipeline
        .replay_cuda_decode_one_token(launch)?
        .map(|result| CudaDecodeTail::new(result, stream.clone()));
    current.release_after(&stream)?;
    Ok(Ok(CudaStepSubmission::new(
        current,
        tail,
        duration + started.elapsed(),
    )))
}

#[cfg(feature = "cuda")]
pub(crate) fn submit_decode_tail<P: Pipeline + ?Sized>(
    pipeline: &mut P,
    seqs: &[&mut Sequence],
    mut tail: CudaDecodeTail,
    duration: Duration,
    lookahead: StepLookahead,
    rng: &Arc<std::sync::Mutex<Isaac64Rng>>,
) -> candle_core::Result<CudaTailSubmission> {
    let started = Instant::now();
    if tail.batch_size()? != seqs.len() || !sampling::can_submit_cuda_token_batch_seqs(seqs) {
        return Ok(CudaTailSubmission::Unsupported(tail));
    }

    let launch_next = lookahead.is_enabled()
        && sampling::can_launch_one_token_lookahead(seqs, pipeline.get_metadata().max_seq_len)
        && tail.launch().is_some();
    let current = if launch_next {
        let launch = tail
            .launch()
            .expect("CUDA decode tail launch was checked above");
        let Some(current) = sampling::try_submit_cuda_token_batch(
            tail.causal_logits()?,
            seqs,
            launch.resident_input(),
            rng,
        )?
        else {
            return Ok(CudaTailSubmission::Unsupported(tail));
        };
        current
    } else {
        let Some(current) =
            sampling::try_submit_cuda_token_batch_owned(tail.causal_logits()?, seqs, rng)?
        else {
            return Ok(CudaTailSubmission::Unsupported(tail));
        };
        tail.take_result();
        return Ok(CudaTailSubmission::Submitted(CudaStepSubmission::new(
            current,
            None,
            duration + started.elapsed(),
        )));
    };

    let stream = tail.stream.clone();
    current.wait_on(&stream)?;
    let ForwardStepResult {
        output: _,
        cuda_decode: Some(launch),
    } = tail.take_result()
    else {
        unreachable!("CUDA decode tail launch disappeared after submission")
    };
    let next_tail = pipeline
        .replay_cuda_decode_one_token(launch)?
        .map(|result| CudaDecodeTail::new(result, stream.clone()));
    current.release_after(&stream)?;
    Ok(CudaTailSubmission::Submitted(CudaStepSubmission::new(
        current,
        next_tail,
        duration + started.elapsed(),
    )))
}

#[cfg(test)]
mod tests {
    use super::{StepCompletion, StepLookahead, StepSubmission};
    use std::time::Duration;

    #[test]
    fn lookahead_is_opt_in() {
        assert!(!StepLookahead::default().is_enabled());
        assert!(StepLookahead::OneToken.is_enabled());
    }

    #[test]
    fn ready_submission_preserves_duration() {
        let duration = Duration::from_micros(37);
        let completion = StepSubmission::ready(duration).into_ready().unwrap();
        assert_eq!(completion.duration(), duration);
        let direct = StepCompletion::ready(duration);
        assert_eq!(direct.duration(), duration);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pending_completion_preserves_row_order_and_cardinality() {
        use super::CudaStepPending;

        let pending = CudaStepPending {
            tail: None,
            duration: Duration::ZERO,
            batch_size: 2,
        };
        let completion = pending.finish(vec![17, 5]).unwrap();
        assert_eq!(completion.token_ids(), &[17, 5]);
        assert!(!completion.has_tail());

        let pending = CudaStepPending {
            tail: None,
            duration: Duration::ZERO,
            batch_size: 2,
        };
        assert!(pending.finish(vec![17]).is_err());

        let mut completion = CudaStepPending {
            tail: None,
            duration: Duration::ZERO,
            batch_size: 1,
        }
        .finish(vec![17])
        .unwrap();
        completion.synchronize_tail().unwrap();
    }
}
