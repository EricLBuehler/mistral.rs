use std::{
    collections::{HashSet, VecDeque},
    fmt::Debug,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver},
        Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
};

use candle_core::{quantized::GgmlDType, DType, Device, DeviceLocation, Result};
use sysinfo::System;

use crate::{IsqCaptureMode, IsqType};

const SIZE_IN_GIB: usize = 1024 * 1024 * 1024;
const LARGE_JOB_THRESHOLD_BYTES: usize = SIZE_IN_GIB;
const HOST_BUDGET_NUMERATOR: usize = 9;
const HOST_BUDGET_DENOMINATOR: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsqConsumer {
    ImmediateLoad,
    RuntimeSwap,
    UqffWrite,
}

impl IsqConsumer {
    fn retains_output(self) -> bool {
        matches!(self, Self::RuntimeSwap | Self::UqffWrite)
    }
}

#[derive(Clone, Debug)]
pub struct IsqRequest {
    pub ty: Option<IsqType>,
    pub device: Device,
    pub has_imatrix: bool,
    pub capture: IsqCaptureMode,
    pub consumer: IsqConsumer,
    pub module_key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsqKernelKind {
    Copy,
    Ggml,
    GgmlImatrix,
    GgmlExpertStack,
    Afq,
    Hqq,
    F8,
    Mxfp4,
    Other,
}

#[derive(Clone, Debug, Default)]
pub struct IsqResourceEstimate {
    pub input_bytes: usize,
    pub host_scratch_bytes: usize,
    pub output_bytes: usize,
    pub large_job: bool,
    pub exclusive_device: bool,
}

impl IsqResourceEstimate {
    pub fn host_peak_bytes(&self) -> usize {
        self.input_bytes
            .saturating_add(self.host_scratch_bytes)
            .saturating_add(self.output_bytes)
    }

    pub fn mark_large(mut self) -> Self {
        self.large_job = self.host_peak_bytes() >= LARGE_JOB_THRESHOLD_BYTES;
        self
    }
}

#[derive(Clone, Debug)]
pub struct IsqPlanParams {
    pub kernel: IsqKernelKind,
    pub source_dtype: DType,
    pub source_device: Device,
    pub target_device: Device,
    pub shape: Vec<usize>,
    pub resources: IsqResourceEstimate,
}

pub struct IsqJobOutput<T> {
    pub value: T,
    _output_permit: Option<IsqOutputPermit>,
}

impl<T> IsqJobOutput<T> {
    pub(crate) fn new(value: T, output_permit: Option<IsqOutputPermit>) -> Self {
        Self {
            value,
            _output_permit: output_permit,
        }
    }

    pub fn ready(value: T) -> Self {
        Self {
            value,
            _output_permit: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IsqExecutorConfig {
    pub worker_threads: usize,
    pub host_budget_bytes: usize,
    pub max_large_jobs: usize,
}

impl IsqExecutorConfig {
    pub fn new(ty: Option<IsqType>) -> Self {
        let worker_threads = isq_worker_threads(ty);
        Self {
            worker_threads,
            host_budget_bytes: default_host_budget(),
            max_large_jobs: worker_threads.clamp(1, 4),
        }
    }

    pub fn with_external_reserved_host_bytes(mut self, bytes: usize) -> Self {
        self.host_budget_bytes = self.host_budget_bytes.saturating_sub(bytes);
        self
    }

    pub fn for_tests(worker_threads: usize, host_budget_bytes: usize) -> Self {
        Self {
            worker_threads: worker_threads.max(1),
            host_budget_bytes,
            max_large_jobs: worker_threads.max(1),
        }
    }
}

pub struct IsqExecutor {
    inner: Arc<ExecutorInner>,
}

impl IsqExecutor {
    pub fn new(config: IsqExecutorConfig) -> Self {
        let inner = Arc::new(ExecutorInner {
            state: Mutex::new(ExecutorState::default()),
            cvar: Condvar::new(),
            config,
            workers: Mutex::new(Vec::new()),
            owners: AtomicUsize::new(1),
        });
        let executor = Self { inner };
        executor.start_workers();
        executor
    }

    pub fn worker_threads(&self) -> usize {
        self.inner.config.worker_threads
    }

    pub fn submit<T, F>(
        &self,
        plan: IsqPlanParams,
        consumer: IsqConsumer,
        job: F,
    ) -> Receiver<Result<IsqJobOutput<T>>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel(1);
        let task = Box::new(move || {
            let result = catch_unwind(AssertUnwindSafe(job));
            let (success, send) = match result {
                Ok(result) => {
                    let success = result.is_ok();
                    let send = Box::new(move |permit| {
                        let result = result.map(|value| IsqJobOutput::new(value, permit));
                        let _ = tx.send(result);
                    }) as JobSend;
                    (success, send)
                }
                Err(_) => {
                    let send = Box::new(move |_| {
                        let _ = tx.send(Err(candle_core::Error::msg("ISQ worker panicked")));
                    }) as JobSend;
                    (false, send)
                }
            };
            JobOutcome { success, send }
        }) as JobTask;
        self.inner.enqueue(QueuedJob {
            plan,
            consumer,
            task,
        });
        rx
    }

    fn start_workers(&self) {
        let mut workers = self.inner.workers.lock().expect("ISQ worker lock poisoned");
        if !workers.is_empty() {
            return;
        }
        for _ in 0..self.inner.config.worker_threads {
            let inner = self.inner.clone();
            workers.push(std::thread::spawn(move || worker_loop(inner)));
        }
    }
}

impl Clone for IsqExecutor {
    fn clone(&self) -> Self {
        self.inner.owners.fetch_add(1, Ordering::Relaxed);
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Debug for IsqExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IsqExecutor")
            .field("worker_threads", &self.worker_threads())
            .field("host_budget_bytes", &self.inner.config.host_budget_bytes)
            .field("max_large_jobs", &self.inner.config.max_large_jobs)
            .finish()
    }
}

impl Drop for IsqExecutor {
    fn drop(&mut self) {
        if self.inner.owners.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        {
            let mut state = self.inner.state.lock().expect("ISQ executor lock poisoned");
            state.shutdown = true;
            self.inner.cvar.notify_all();
        }
        let mut workers = self.inner.workers.lock().expect("ISQ worker lock poisoned");
        for worker in workers.drain(..) {
            let _ = worker.join();
        }
    }
}

type JobSend = Box<dyn FnOnce(Option<IsqOutputPermit>) + Send>;
type JobTask = Box<dyn FnOnce() -> JobOutcome + Send>;

struct JobOutcome {
    success: bool,
    send: JobSend,
}

struct QueuedJob {
    plan: IsqPlanParams,
    consumer: IsqConsumer,
    task: JobTask,
}

struct RunningJob {
    plan: IsqPlanParams,
    consumer: IsqConsumer,
}

struct ExecutorInner {
    state: Mutex<ExecutorState>,
    cvar: Condvar,
    config: IsqExecutorConfig,
    workers: Mutex<Vec<JoinHandle<()>>>,
    owners: AtomicUsize,
}

#[derive(Default)]
struct ExecutorState {
    pending: VecDeque<QueuedJob>,
    held_input_bytes: usize,
    active_host_scratch: usize,
    retained_output: usize,
    active_large_jobs: usize,
    active_exclusive_devices: HashSet<DeviceLocation>,
    shutdown: bool,
}

impl ExecutorInner {
    fn enqueue(&self, job: QueuedJob) {
        let mut state = self.state.lock().expect("ISQ executor lock poisoned");
        while !can_enqueue(&state, &self.config, &job) {
            state = self.cvar.wait(state).expect("ISQ executor lock poisoned");
        }
        state.held_input_bytes = state
            .held_input_bytes
            .saturating_add(job.plan.resources.input_bytes);
        state.pending.push_back(job);
        self.cvar.notify_all();
    }

    fn release_output(self: &Arc<Self>, bytes: usize) {
        let mut state = self.state.lock().expect("ISQ executor lock poisoned");
        state.retained_output = state.retained_output.saturating_sub(bytes);
        self.cvar.notify_all();
    }
}

pub(crate) struct IsqOutputPermit {
    inner: Arc<ExecutorInner>,
    bytes: usize,
}

impl Drop for IsqOutputPermit {
    fn drop(&mut self) {
        self.inner.release_output(self.bytes);
    }
}

fn worker_loop(inner: Arc<ExecutorInner>) {
    unsafe {
        crate::set_isq_thread_affinity();
    }
    loop {
        let Some(job) = next_job(&inner) else {
            return;
        };
        let running = RunningJob {
            plan: job.plan.clone(),
            consumer: job.consumer,
        };
        let outcome = (job.task)();
        let permit = finish_job(&inner, running, outcome.success);
        (outcome.send)(permit);
    }
}

fn next_job(inner: &Arc<ExecutorInner>) -> Option<QueuedJob> {
    let mut state = inner.state.lock().expect("ISQ executor lock poisoned");
    loop {
        if state.shutdown && state.pending.is_empty() {
            return None;
        }
        if let Some(index) = state
            .pending
            .iter()
            .position(|job| can_start(&state, &inner.config, job))
        {
            let job = state.pending.remove(index).expect("pending job exists");
            reserve_job(&mut state, &job);
            return Some(job);
        }
        state = inner.cvar.wait(state).expect("ISQ executor lock poisoned");
    }
}

fn can_enqueue(state: &ExecutorState, config: &IsqExecutorConfig, job: &QueuedJob) -> bool {
    let resources = &job.plan.resources;
    let host_next = state
        .held_input_bytes
        .saturating_add(state.active_host_scratch)
        .saturating_add(state.retained_output)
        .saturating_add(resources.input_bytes);
    let no_host_work =
        state.held_input_bytes == 0 && state.active_host_scratch == 0 && state.retained_output == 0;
    host_next <= config.host_budget_bytes || no_host_work
}

fn can_start(state: &ExecutorState, config: &IsqExecutorConfig, job: &QueuedJob) -> bool {
    let resources = &job.plan.resources;
    let host_next = state
        .held_input_bytes
        .saturating_add(state.active_host_scratch)
        .saturating_add(state.retained_output)
        .saturating_add(resources.host_scratch_bytes)
        .saturating_add(resources.output_bytes);
    let no_active_host_work = state.active_host_scratch == 0 && state.retained_output == 0;
    if host_next > config.host_budget_bytes && !no_active_host_work {
        return false;
    }
    if resources.large_job && state.active_large_jobs >= config.max_large_jobs {
        return false;
    }
    if resources.exclusive_device {
        if let Some(key) = device_key(&job.plan.target_device) {
            if state.active_exclusive_devices.contains(&key) {
                return false;
            }
        }
    }
    true
}

fn reserve_job(state: &mut ExecutorState, job: &QueuedJob) {
    let resources = &job.plan.resources;
    state.active_host_scratch = state
        .active_host_scratch
        .saturating_add(resources.host_scratch_bytes);
    state.retained_output = state.retained_output.saturating_add(resources.output_bytes);
    if resources.large_job {
        state.active_large_jobs += 1;
    }
    if resources.exclusive_device {
        if let Some(key) = device_key(&job.plan.target_device) {
            state.active_exclusive_devices.insert(key);
        }
    }
}

fn finish_job(
    inner: &Arc<ExecutorInner>,
    running: RunningJob,
    success: bool,
) -> Option<IsqOutputPermit> {
    let resources = &running.plan.resources;
    let retain_output = success && running.consumer.retains_output();
    let mut state = inner.state.lock().expect("ISQ executor lock poisoned");
    state.held_input_bytes = state.held_input_bytes.saturating_sub(resources.input_bytes);
    state.active_host_scratch = state
        .active_host_scratch
        .saturating_sub(resources.host_scratch_bytes);
    if resources.large_job {
        state.active_large_jobs = state.active_large_jobs.saturating_sub(1);
    }
    if resources.exclusive_device {
        if let Some(key) = device_key(&running.plan.target_device) {
            state.active_exclusive_devices.remove(&key);
        }
    }
    if !retain_output {
        state.retained_output = state.retained_output.saturating_sub(resources.output_bytes);
    }
    inner.cvar.notify_all();
    drop(state);
    retain_output.then(|| IsqOutputPermit {
        inner: inner.clone(),
        bytes: resources.output_bytes,
    })
}

fn default_host_budget() -> usize {
    let mut sys = System::new();
    sys.refresh_memory();
    let available = usize::try_from(sys.available_memory()).unwrap_or(usize::MAX);
    available
        .saturating_mul(HOST_BUDGET_NUMERATOR)
        .saturating_div(HOST_BUDGET_DENOMINATOR)
}

pub fn isq_worker_threads(ty: Option<IsqType>) -> usize {
    if std::env::var("MISTRALRS_ISQ_SINGLETHREAD").is_ok() {
        return 1;
    }
    if let Some(ty) = ty.and_then(|ty| ty.get_max_isq_cpu_threads()) {
        return usize::from(ty);
    }
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

pub fn elem_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

pub fn tensor_bytes(shape: &[usize], dtype: DType) -> usize {
    elem_count(shape).saturating_mul(dtype.size_in_bytes())
}

pub fn ggml_output_bytes(shape: &[usize], ty: IsqType) -> Option<usize> {
    let dtype = ggml_dtype_for_estimate(ty)?;
    let elements = elem_count(shape);
    Some(
        elements
            .div_ceil(dtype.block_size())
            .saturating_mul(dtype.type_size()),
    )
}

pub fn conservative_plan(
    kernel: IsqKernelKind,
    source_dtype: DType,
    source_device: Device,
    shape: Vec<usize>,
    output_bytes: usize,
    exclusive_device: bool,
) -> IsqPlanParams {
    let source_bytes = tensor_bytes(&shape, source_dtype);
    let host_scratch_bytes = source_bytes.max(elem_count(&shape).saturating_mul(4));
    IsqPlanParams {
        kernel,
        source_dtype,
        source_device: source_device.clone(),
        target_device: source_device,
        shape,
        resources: IsqResourceEstimate {
            input_bytes: 0,
            host_scratch_bytes,
            output_bytes,
            large_job: false,
            exclusive_device,
        }
        .mark_large(),
    }
}

pub fn plan_weight_isq(
    source_dtype: DType,
    source_device: Device,
    shape: Vec<usize>,
    request: &IsqRequest,
    dequantize_before_quantize: bool,
) -> IsqPlanParams {
    let elements = elem_count(&shape);
    let source_bytes = tensor_bytes(&shape, source_dtype);
    let output_bytes = request
        .ty
        .map(|ty| estimate_output_bytes(&shape, source_dtype, ty))
        .unwrap_or(source_bytes);
    let kernel = kernel_for(request, shape.len());
    let ggml = matches!(
        kernel,
        IsqKernelKind::Ggml | IsqKernelKind::GgmlImatrix | IsqKernelKind::GgmlExpertStack
    );
    let host_scratch_bytes = if request.ty.is_none() {
        0
    } else if ggml || dequantize_before_quantize {
        elements.saturating_mul(4)
    } else {
        source_bytes
    };
    let input_bytes = if request.consumer == IsqConsumer::ImmediateLoad {
        source_bytes
    } else {
        0
    };
    let exclusive_device = !request.device.is_cpu()
        && matches!(
            kernel,
            IsqKernelKind::Afq | IsqKernelKind::Hqq | IsqKernelKind::F8 | IsqKernelKind::Mxfp4
        );
    IsqPlanParams {
        kernel,
        source_dtype,
        source_device,
        target_device: request.device.clone(),
        shape,
        resources: IsqResourceEstimate {
            input_bytes,
            host_scratch_bytes,
            output_bytes,
            large_job: false,
            exclusive_device,
        }
        .mark_large(),
    }
}

pub fn estimate_output_bytes(shape: &[usize], source_dtype: DType, ty: IsqType) -> usize {
    if let Some(bytes) = ggml_output_bytes(shape, ty) {
        return bytes;
    }
    let source_bytes = tensor_bytes(shape, source_dtype);
    match ty {
        IsqType::F8Q8 => elem_count(shape).div_ceil(32).saturating_mul(33),
        IsqType::F8E4M3 => elem_count(shape),
        _ => source_bytes
            .checked_div(ty.pack_factor(source_dtype).max(1))
            .unwrap_or(source_bytes)
            .max(source_dtype.size_in_bytes()),
    }
}

fn kernel_for(request: &IsqRequest, rank: usize) -> IsqKernelKind {
    match request.ty {
        None => IsqKernelKind::Copy,
        Some(ty) if is_ggml_isq(ty) && request.has_imatrix && rank == 3 => {
            IsqKernelKind::GgmlExpertStack
        }
        Some(ty) if is_ggml_isq(ty) && request.has_imatrix => IsqKernelKind::GgmlImatrix,
        Some(ty) if is_ggml_isq(ty) => IsqKernelKind::Ggml,
        Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
            IsqKernelKind::Afq
        }
        Some(IsqType::HQQ4 | IsqType::HQQ8) => IsqKernelKind::Hqq,
        Some(IsqType::F8E4M3 | IsqType::F8Q8) => IsqKernelKind::F8,
        Some(IsqType::MXFP4) => IsqKernelKind::Mxfp4,
        Some(_) => IsqKernelKind::Other,
    }
}

fn is_ggml_isq(ty: IsqType) -> bool {
    matches!(
        ty,
        IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1
    )
}

fn ggml_dtype_for_estimate(ty: IsqType) -> Option<GgmlDType> {
    Some(match ty {
        IsqType::Q2K => GgmlDType::Q2K,
        IsqType::Q3K => GgmlDType::Q3K,
        IsqType::Q4K => GgmlDType::Q4K,
        IsqType::Q4_0 => GgmlDType::Q4_0,
        IsqType::Q4_1 => GgmlDType::Q4_1,
        IsqType::Q5K => GgmlDType::Q5K,
        IsqType::Q5_0 => GgmlDType::Q5_0,
        IsqType::Q5_1 => GgmlDType::Q5_1,
        IsqType::Q6K => GgmlDType::Q6K,
        IsqType::Q8K => GgmlDType::Q8K,
        IsqType::Q8_0 => GgmlDType::Q8_0,
        IsqType::Q8_1 => GgmlDType::Q8_1,
        _ => return None,
    })
}

fn device_key(device: &Device) -> Option<DeviceLocation> {
    if device.is_cpu() {
        None
    } else {
        Some(device.location())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex as StdMutex,
    };
    use std::time::Duration;

    use super::*;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());
    const TEST_RECV_TIMEOUT: Duration = Duration::from_secs(1);

    fn plan(bytes: usize) -> IsqPlanParams {
        IsqPlanParams {
            kernel: IsqKernelKind::Ggml,
            source_dtype: DType::BF16,
            source_device: Device::Cpu,
            target_device: Device::Cpu,
            shape: vec![bytes / 2],
            resources: IsqResourceEstimate {
                host_scratch_bytes: bytes,
                output_bytes: bytes,
                ..Default::default()
            }
            .mark_large(),
        }
    }

    fn input_plan(bytes: usize) -> IsqPlanParams {
        let mut plan = plan(0);
        plan.resources.input_bytes = bytes;
        plan.resources.large_job = false;
        plan
    }

    fn queued_job(plan: IsqPlanParams, consumer: IsqConsumer) -> QueuedJob {
        QueuedJob {
            plan,
            consumer,
            task: Box::new(|| JobOutcome {
                success: true,
                send: Box::new(|_| {}),
            }),
        }
    }

    #[test]
    fn q8_0_estimate_uses_block_size() {
        let shape = vec![2_415_919_104usize];
        assert_eq!(
            ggml_output_bytes(&shape, IsqType::Q8_0),
            Some(2_566_914_048)
        );
    }

    #[test]
    fn executor_waits_for_budget_release() {
        let executor = IsqExecutor::new(IsqExecutorConfig::for_tests(2, 100));
        let started = Arc::new(AtomicUsize::new(0));
        let first_started = started.clone();
        let rx1 = executor.submit(plan(60), IsqConsumer::UqffWrite, move || {
            first_started.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(100));
            Ok(1usize)
        });
        let second_started = started.clone();
        let rx2 = executor.submit(plan(60), IsqConsumer::UqffWrite, move || {
            second_started.fetch_add(1, Ordering::SeqCst);
            Ok(2usize)
        });

        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(started.load(Ordering::SeqCst), 1);
        let first = rx1.recv().unwrap().unwrap();
        assert_eq!(started.load(Ordering::SeqCst), 1);
        drop(first);
        assert_eq!(rx2.recv().unwrap().unwrap().value, 2);
    }

    #[test]
    fn executor_blocks_submit_for_queued_input_bytes() {
        let executor = IsqExecutor::new(IsqExecutorConfig::for_tests(2, 100));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let rx1 = executor.submit(input_plan(70), IsqConsumer::ImmediateLoad, move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(1usize)
        });
        started_rx.recv_timeout(TEST_RECV_TIMEOUT).unwrap();

        let second = queued_job(input_plan(70), IsqConsumer::ImmediateLoad);
        {
            let state = executor.inner.state.lock().unwrap();
            assert!(!can_enqueue(&state, &executor.inner.config, &second));
        }

        release_tx.send(()).unwrap();
        assert_eq!(
            rx1.recv_timeout(TEST_RECV_TIMEOUT).unwrap().unwrap().value,
            1
        );
        {
            let state = executor.inner.state.lock().unwrap();
            assert!(can_enqueue(&state, &executor.inner.config, &second));
        }
        let rx2 = executor.submit(input_plan(70), IsqConsumer::ImmediateLoad, move || {
            Ok(2usize)
        });
        assert_eq!(
            rx2.recv_timeout(TEST_RECV_TIMEOUT).unwrap().unwrap().value,
            2
        );
    }

    #[test]
    fn retained_out_of_order_output_blocks_later_jobs() {
        let executor = IsqExecutor::new(IsqExecutorConfig::for_tests(2, 170));
        let started = Arc::new(AtomicUsize::new(0));
        let mut output_plan = plan(80);
        output_plan.resources.host_scratch_bytes = 0;
        let first_started = started.clone();
        let rx1 = executor.submit(output_plan.clone(), IsqConsumer::UqffWrite, move || {
            first_started.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(120));
            Ok(1usize)
        });
        let second_started = started.clone();
        let rx2 = executor.submit(output_plan.clone(), IsqConsumer::UqffWrite, move || {
            second_started.fetch_add(1, Ordering::SeqCst);
            Ok(2usize)
        });
        let third_started = started.clone();
        let rx3 = executor.submit(output_plan, IsqConsumer::UqffWrite, move || {
            third_started.fetch_add(1, Ordering::SeqCst);
            Ok(3usize)
        });

        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(started.load(Ordering::SeqCst), 2);
        let second = rx2.recv().unwrap().unwrap();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(started.load(Ordering::SeqCst), 2);
        let first = rx1.recv().unwrap().unwrap();
        drop(second);
        drop(first);
        assert_eq!(rx3.recv().unwrap().unwrap().value, 3);
    }

    #[test]
    fn singlethread_env_sets_one_worker() {
        let _guard = ENV_LOCK.lock().unwrap();
        let old = std::env::var_os("MISTRALRS_ISQ_SINGLETHREAD");
        std::env::set_var("MISTRALRS_ISQ_SINGLETHREAD", "1");
        assert_eq!(
            IsqExecutorConfig::new(Some(IsqType::Q8_0)).worker_threads,
            1
        );
        if let Some(old) = old {
            std::env::set_var("MISTRALRS_ISQ_SINGLETHREAD", old);
        } else {
            std::env::remove_var("MISTRALRS_ISQ_SINGLETHREAD");
        }
    }
}
