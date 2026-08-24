use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    thread::{self, JoinHandle},
};

use candle_core::{Error, Result};
use tokio::sync::{mpsc, oneshot};

#[cfg(feature = "cuda")]
use crate::pipeline::sampling::CudaGreedyBatchSubmission;

const COMPLETION_QUEUE_CAPACITY: usize = 2;
const COMPLETION_CHANNEL_CLOSED: &str = "CUDA decode completion worker stopped";
#[cfg(feature = "cuda")]
const COMPLETION_THREAD_NAME: &str = "mistralrs-cuda-decode-completion";

trait CompletionSubmission: Send + 'static {
    fn complete(self) -> Result<Vec<u32>>;
}

#[cfg(feature = "cuda")]
impl CompletionSubmission for CudaGreedyBatchSubmission {
    fn complete(self) -> Result<Vec<u32>> {
        CudaGreedyBatchSubmission::complete(self)
    }
}

struct CompletionJob<S> {
    submission: S,
    result_tx: oneshot::Sender<Result<Vec<u32>>>,
}

pub(crate) struct CudaDecodeCompletion {
    result_rx: oneshot::Receiver<Result<Vec<u32>>>,
}

impl Future for CudaDecodeCompletion {
    type Output = Result<Vec<u32>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.result_rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => {
                Poll::Ready(Err(Error::Msg(COMPLETION_CHANNEL_CLOSED.to_string())))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

struct CompletionWorker<S: CompletionSubmission> {
    job_tx: Option<mpsc::Sender<CompletionJob<S>>>,
    thread: Option<JoinHandle<()>>,
}

impl<S: CompletionSubmission> CompletionWorker<S> {
    fn new(thread_name: &str) -> Result<Self> {
        let (job_tx, mut job_rx) = mpsc::channel::<CompletionJob<S>>(COMPLETION_QUEUE_CAPACITY);
        let thread = thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || {
                while let Some(job) = job_rx.blocking_recv() {
                    let result = job.submission.complete();
                    let _ = job.result_tx.send(result);
                }
            })
            .map_err(|err| {
                Error::Msg(format!(
                    "failed to start CUDA decode completion worker: {err}"
                ))
            })?;
        Ok(Self {
            job_tx: Some(job_tx),
            thread: Some(thread),
        })
    }

    async fn submit(&self, submission: S) -> Result<CudaDecodeCompletion> {
        let (result_tx, result_rx) = oneshot::channel();
        let job = CompletionJob {
            submission,
            result_tx,
        };
        self.job_tx
            .as_ref()
            .ok_or_else(|| Error::Msg(COMPLETION_CHANNEL_CLOSED.to_string()))?
            .send(job)
            .await
            .map_err(|_| Error::Msg(COMPLETION_CHANNEL_CLOSED.to_string()))?;
        Ok(CudaDecodeCompletion { result_rx })
    }
}

impl<S: CompletionSubmission> Drop for CompletionWorker<S> {
    fn drop(&mut self) {
        drop(self.job_tx.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct CudaDecodeCompletionWorker {
    worker: CompletionWorker<CudaGreedyBatchSubmission>,
}

#[cfg(feature = "cuda")]
impl CudaDecodeCompletionWorker {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            worker: CompletionWorker::new(COMPLETION_THREAD_NAME)?,
        })
    }

    pub(crate) async fn submit(
        &self,
        submission: CudaGreedyBatchSubmission,
    ) -> Result<CudaDecodeCompletion> {
        self.worker.submit(submission).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;

    struct TestSubmission {
        completed: Arc<AtomicUsize>,
        tokens: Vec<u32>,
    }

    impl CompletionSubmission for TestSubmission {
        fn complete(self) -> Result<Vec<u32>> {
            self.completed.fetch_add(1, Ordering::Relaxed);
            Ok(self.tokens)
        }
    }

    #[tokio::test]
    async fn returns_completion_through_oneshot() {
        let completed = Arc::new(AtomicUsize::new(0));
        let worker = CompletionWorker::new("cuda-decode-completion-test").unwrap();
        let completion = worker
            .submit(TestSubmission {
                completed: completed.clone(),
                tokens: vec![11, 29],
            })
            .await
            .unwrap();

        assert_eq!(completion.await.unwrap(), vec![11, 29]);
        assert_eq!(completed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn dropped_receivers_do_not_cancel_submissions() {
        let completed = Arc::new(AtomicUsize::new(0));
        let worker = CompletionWorker::new("cuda-decode-completion-drop-test").unwrap();
        let completion = worker
            .submit(TestSubmission {
                completed: completed.clone(),
                tokens: vec![7],
            })
            .await
            .unwrap();
        drop(completion);
        drop(worker);

        assert_eq!(completed.load(Ordering::Relaxed), 1);
    }
}
