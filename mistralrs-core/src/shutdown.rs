use std::{
    cell::Cell,
    collections::HashSet,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Condvar, Mutex, RwLock},
    thread::JoinHandle,
};

use tokio::sync::mpsc::Sender;

use crate::{EngineInstance, MistralRs, MistralRsError, Request};

/// Safe shutdown diagnostics. Native panic payloads, model paths and request bodies are omitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MistralRsShutdownFailure {
    TerminateChannelClosed,
    EnginePanicked,
    ResourcesPanicked,
    RegistryPoisoned,
    ShutdownPanicked,
    EngineThreadCannotJoin,
}

impl std::fmt::Display for MistralRsShutdownFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::TerminateChannelClosed => "engine termination channel closed",
            Self::EnginePanicked => "engine thread panicked",
            Self::ResourcesPanicked => "engine resources panicked during release",
            Self::RegistryPoisoned => "engine registry lock was poisoned",
            Self::ShutdownPanicked => "engine shutdown owner panicked",
            Self::EngineThreadCannotJoin => "an engine thread cannot join engine shutdown",
        })
    }
}

impl std::error::Error for MistralRsShutdownFailure {}

/// All close failures retained after every available engine thread has been joined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MistralRsShutdownError(Arc<[MistralRsShutdownFailure]>);

impl MistralRsShutdownError {
    pub fn failures(&self) -> &[MistralRsShutdownFailure] {
        &self.0
    }
}

impl std::fmt::Display for MistralRsShutdownError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "mistral.rs engine shutdown failed")?;
        for failure in self.failures() {
            write!(formatter, "; {failure}")?;
        }
        Ok(())
    }
}

impl std::error::Error for MistralRsShutdownError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.failures().first().map(|failure| failure as _)
    }
}

type ShutdownResult = Result<(), MistralRsShutdownError>;

#[derive(Default)]
enum Phase {
    #[default]
    Running,
    Joining,
    Finished(ShutdownResult),
}

#[derive(Default)]
struct State {
    phase: Phase,
    admitted: usize,
    failures: Vec<MistralRsShutdownFailure>,
}

/// Serializes explicit shutdown and waits for previously admitted model management calls.
#[derive(Default)]
pub(crate) struct Shutdown {
    state: Mutex<State>,
    changed: Condvar,
}

pub(crate) struct Admission<'a>(&'a Shutdown);

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().expect("engine shutdown state lock");
        state.admitted -= 1;
        self.0.changed.notify_all();
    }
}

impl Shutdown {
    pub(crate) fn enter(&self) -> Result<Admission<'_>, MistralRsError> {
        ensure_not_engine_thread().map_err(MistralRsError::Shutdown)?;
        let mut state = self.state.lock().expect("engine shutdown state lock");
        if !matches!(state.phase, Phase::Running) {
            return Err(MistralRsError::ShuttingDown);
        }
        if !state.failures.is_empty() {
            return Err(MistralRsError::Shutdown(MistralRsShutdownError(
                state.failures.clone().into(),
            )));
        }
        state.admitted += 1;
        Ok(Admission(self))
    }

    pub(crate) fn record(&self, failures: Vec<MistralRsShutdownFailure>) -> ShutdownResult {
        let result = outcome(failures.clone());
        self.state
            .lock()
            .expect("engine shutdown state lock")
            .failures
            .extend(failures);
        result
    }

    pub(crate) fn run(
        &self,
        close: impl FnOnce() -> Vec<MistralRsShutdownFailure>,
    ) -> ShutdownResult {
        // Check before waiting on another shutdown: that caller may already be joining us.
        ensure_not_engine_thread()?;
        let mut state = self.state.lock().expect("engine shutdown state lock");
        loop {
            match &state.phase {
                Phase::Finished(result) => return result.clone(),
                Phase::Joining => {
                    state = self
                        .changed
                        .wait(state)
                        .expect("engine shutdown state lock")
                }
                Phase::Running => {
                    state.phase = Phase::Joining;
                    self.changed.notify_all();
                    break;
                }
            }
        }
        while state.admitted != 0 {
            state = self
                .changed
                .wait(state)
                .expect("engine shutdown state lock");
        }
        drop(state);
        let failures = catch_unwind(AssertUnwindSafe(close))
            .unwrap_or_else(|_| vec![MistralRsShutdownFailure::ShutdownPanicked]);
        let mut state = self.state.lock().expect("engine shutdown state lock");
        state.failures.extend(failures);
        let result = outcome(std::mem::take(&mut state.failures));
        state.phase = Phase::Finished(result.clone());
        self.changed.notify_all();
        result
    }

    #[cfg(test)]
    pub(crate) fn wait_for_shutdown(&self, timeout: std::time::Duration) -> bool {
        let state = self.state.lock().expect("engine shutdown state lock");
        let (state, _) = self
            .changed
            .wait_timeout_while(state, timeout, |state| {
                matches!(state.phase, Phase::Running)
            })
            .expect("engine shutdown state lock");
        !matches!(state.phase, Phase::Running)
    }
}

pub(super) fn ensure_not_engine_thread() -> ShutdownResult {
    if ENGINE_THREAD.with(Cell::get) {
        Err(MistralRsShutdownError(
            vec![MistralRsShutdownFailure::EngineThreadCannotJoin].into(),
        ))
    } else {
        Ok(())
    }
}

/// Reuses the existing reload set to reserve a model while its old worker is being joined.
/// No registry lock is held across a native join or an asynchronous model load.
pub(super) struct ModelOperation<'a> {
    models: &'a RwLock<HashSet<String>>,
    id: String,
}

impl Drop for ModelOperation<'_> {
    fn drop(&mut self) {
        self.models
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.id);
    }
}

thread_local! {
    // Execution identity only, never model residency or a process-wide engine owner.
    static ENGINE_THREAD: Cell<bool> = const { Cell::new(false) };
}

pub(crate) struct EngineThreadScope(bool);

impl EngineThreadScope {
    pub(crate) fn enter() -> Self {
        Self(ENGINE_THREAD.replace(true))
    }
}

impl Drop for EngineThreadScope {
    fn drop(&mut self) {
        ENGINE_THREAD.set(self.0);
    }
}

pub(crate) fn engine_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(candle_core::utils::get_num_threads())
        .on_thread_start(|| {
            ENGINE_THREAD.set(true);
            candle_core::utils::set_thread_affinity();
        })
        .on_thread_stop(|| ENGINE_THREAD.set(false))
        .build()
}

/// Holds all native resources until the actual worker thread, including its local runtime,
/// has exited. Generic resources let offline tests exercise the same join implementation.
pub(crate) struct RetiringEngine<T> {
    sender: Sender<Request>,
    worker: Option<JoinHandle<()>>,
    resources: T,
}

impl<T> RetiringEngine<T> {
    pub(crate) fn new(
        sender: Sender<Request>,
        worker: Option<JoinHandle<()>>,
        resources: T,
    ) -> Self {
        Self {
            sender,
            worker,
            resources,
        }
    }
}

pub(crate) fn join_engines<T>(engines: Vec<RetiringEngine<T>>) -> Vec<MistralRsShutdownFailure> {
    let mut failures = Vec::new();
    // Signal every engine before joining any of them. A closed channel is still followed by
    // the real join, so an already panicked worker never causes later engines to be skipped.
    for engine in &engines {
        if engine
            .worker
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
            && futures::executor::block_on(engine.sender.send(Request::Terminate)).is_err()
        {
            failures.push(MistralRsShutdownFailure::TerminateChannelClosed);
        }
    }
    for engine in engines {
        let RetiringEngine {
            sender,
            worker,
            resources,
        } = engine;
        drop(sender);
        if worker.is_some_and(|worker| worker.join().is_err()) {
            failures.push(MistralRsShutdownFailure::EnginePanicked);
        }
        if catch_unwind(AssertUnwindSafe(|| drop(resources))).is_err() {
            failures.push(MistralRsShutdownFailure::ResourcesPanicked);
        }
    }
    failures
}

fn into_retiring(mut engine: EngineInstance) -> RetiringEngine<EngineInstance> {
    // Keep the complete owner alive, including upstream's CUDA-graph cleanup destructor.
    RetiringEngine::new(engine.sender.clone(), engine.engine_handler.take(), engine)
}

pub(crate) fn retire_engine(engine: EngineInstance) -> Vec<MistralRsShutdownFailure> {
    join_engines(vec![into_retiring(engine)])
}

fn outcome(failures: Vec<MistralRsShutdownFailure>) -> ShutdownResult {
    if failures.is_empty() {
        Ok(())
    } else {
        Err(MistralRsShutdownError(failures.into()))
    }
}

impl MistralRs {
    pub(super) fn begin_model_operation(
        &self,
        model_id: &str,
    ) -> Result<ModelOperation<'_>, MistralRsError> {
        let mut models = self
            .reloading_models
            .write()
            .map_err(|_| MistralRsError::EnginePoisoned)?;
        if !models.insert(model_id.to_string()) {
            return Err(MistralRsError::ModelReloading(model_id.to_string()));
        }
        Ok(ModelOperation {
            models: &self.reloading_models,
            id: model_id.to_string(),
        })
    }

    /// Transfer a newly created worker to the registry, or join it before returning failure.
    /// The per-model operation prevents a concurrent remove/reload from replacing this owner.
    pub(super) fn insert_engine(
        &self,
        model_id: &str,
        engine: EngineInstance,
    ) -> Result<bool, MistralRsError> {
        let error = match self.engines.write() {
            Ok(mut engines) => {
                if !engines.contains_key(model_id) {
                    engines.insert(model_id.to_string(), engine);
                    return Ok(engines.len() == 1);
                }
                MistralRsError::ModelAlreadyLoaded(model_id.to_string())
            }
            Err(poisoned) => {
                drop(poisoned);
                MistralRsError::EnginePoisoned
            }
        };
        self.shutdown
            .record(retire_engine(engine))
            .map_err(MistralRsError::Shutdown)?;
        Err(error)
    }

    /// Close admission, wait for admitted model-management calls, terminate all engines and
    /// join their actual threads. Async callers must use a blocking worker. Repeated and
    /// concurrent callers receive the same retained result; engine-thread calls are rejected.
    /// This does not rely on sender closure or the best-effort Drop termination signal.
    pub fn shutdown_blocking(&self) -> ShutdownResult {
        self.shutdown.run(|| {
            let mut failures = Vec::new();
            let engines = {
                let mut registry = self.engines.write().unwrap_or_else(|poisoned| {
                    failures.push(MistralRsShutdownFailure::RegistryPoisoned);
                    poisoned.into_inner()
                });
                std::mem::take(&mut *registry)
            };
            let engines = engines.into_values().map(into_retiring).collect();
            failures.extend(join_engines(engines));
            failures
        })
    }
}
