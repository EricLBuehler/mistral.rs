use anyhow::Context;
use candle_core::{DType, Device};
use core::ffi::c_char;
use interprocess::local_socket::traits::{Listener, Stream};
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
pub use mistralrs_quant::distributed::{use_nccl, use_ring};
use mistralrs_quant::{RingConfig, ShardedVarBuilder};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::env;
use std::future::Future;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::Command;
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::Sender;
use tracing::info;

use crate::device_map::DeviceMapper;
use crate::paged_attention::ModelConfigLike;
use crate::pipeline::{DeviceMappedModelLoader, IsqModelLoader};
use crate::utils::varbuilder_utils::{self, DeviceForLoadTensor};
use crate::{DeviceMapSetting, IsqOrganization, ModelPaths, Request};

pub(crate) const IS_DAEMON_FLAG: &str = "__MISTRALRS_DAEMON_INTERNAL";
const MISTRALRS_MN_GLOBAL_WORLD_SIZE: &str = "MISTRALRS_MN_GLOBAL_WORLD_SIZE";
const MISTRALRS_MN_LOCAL_WORLD_SIZE: &str = "MISTRALRS_MN_LOCAL_WORLD_SIZE";
const WORKER_READY_CONNECT_ATTEMPTS: usize = 200;
const WORKER_READY_CONNECT_SLEEP_MS: u64 = 50;

static TP_SESSION: OnceLock<Mutex<TensorParallelSession>> = OnceLock::new();

struct TensorParallelSession {
    ids: Vec<mistralrs_quant::Id>,
    next_model_index: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct TensorParallelism {
    world_size: Option<usize>,
}

impl TensorParallelism {
    pub(crate) fn disabled() -> Self {
        Self { world_size: None }
    }

    pub(crate) fn enabled(world_size: usize) -> Self {
        Self {
            world_size: Some(world_size),
        }
    }

    pub(crate) fn is_enabled(self) -> bool {
        self.world_size.is_some_and(|world_size| world_size > 1)
    }

    pub(crate) fn world_size(self) -> Option<usize> {
        self.world_size
    }
}

pub(crate) fn resolve_tensor_parallelism(
    model_config: &dyn ModelConfigLike,
    use_nccl: bool,
    write_uqff: bool,
) -> anyhow::Result<TensorParallelism> {
    if write_uqff {
        return Ok(TensorParallelism::disabled());
    }

    let use_ring = use_ring();
    if !use_nccl && !use_ring {
        return Ok(TensorParallelism::disabled());
    }

    let Some(requested_world_size) = requested_tensor_parallel_size(use_nccl, use_ring)? else {
        return Ok(TensorParallelism::disabled());
    };
    if requested_world_size <= 1 {
        return Ok(TensorParallelism::disabled());
    }

    if tensor_parallel_size_is_explicit(use_ring) {
        validate_model_tensor_parallelism(model_config, requested_world_size).with_context(
            || {
                format!(
                    "Explicit tensor parallel size {requested_world_size} is incompatible with this model"
                )
            },
        )?;
        return Ok(TensorParallelism::enabled(requested_world_size));
    }

    match select_compatible_tensor_parallel_size(model_config, requested_world_size)? {
        Some(world_size) if world_size == requested_world_size => {
            Ok(TensorParallelism::enabled(world_size))
        }
        Some(world_size) => {
            info!(
                "Auto tensor parallel size {requested_world_size} is incompatible with this model; using tensor parallel size {world_size}."
            );
            Ok(TensorParallelism::enabled(world_size))
        }
        None => {
            info!(
                "Auto tensor parallel size {requested_world_size} is incompatible with this model; disabling tensor parallelism."
            );
            Ok(TensorParallelism::disabled())
        }
    }
}

fn requested_tensor_parallel_size(use_nccl: bool, use_ring: bool) -> anyhow::Result<Option<usize>> {
    if use_ring {
        return Ok(Some(RingConfig::load().world_size));
    }
    if use_nccl {
        return Ok(Some(
            mistralrs_quant::distributed::get_global_tp_size_from_devices()?,
        ));
    }
    Ok(None)
}

fn tensor_parallel_size_is_explicit(use_ring: bool) -> bool {
    use_ring
        || env::var(MISTRALRS_MN_GLOBAL_WORLD_SIZE).is_ok()
        || env::var(MISTRALRS_MN_LOCAL_WORLD_SIZE).is_ok()
}

fn validate_model_tensor_parallelism(
    model_config: &dyn ModelConfigLike,
    world_size: usize,
) -> anyhow::Result<()> {
    for layer_idx in 0..model_config.num_layers() {
        let spec = model_config.attention_layer_spec(layer_idx);
        mistralrs_quant::validate_tp_head_layout(spec.q_heads, spec.kv_heads, world_size)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("Layer {layer_idx}"))?;
    }
    Ok(())
}

fn select_compatible_tensor_parallel_size(
    model_config: &dyn ModelConfigLike,
    requested_world_size: usize,
) -> anyhow::Result<Option<usize>> {
    if requested_world_size <= 1 {
        return Ok(None);
    }
    for world_size in (2..=requested_world_size).rev() {
        if validate_model_tensor_parallelism(model_config, world_size).is_ok() {
            return Ok(Some(world_size));
        }
    }
    Ok(None)
}

pub fn is_daemon() -> bool {
    if cfg!(feature = "cuda") && !cfg!(feature = "ring") {
        std::env::var(IS_DAEMON_FLAG).is_ok()
    } else if use_ring() {
        !RingConfig::load().is_master_rank()
    } else {
        false
    }
}

pub fn nccl_daemon_replicator(request_sender: Sender<Request>) {
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            use interprocess::local_socket::traits::Stream;
            use interprocess::local_socket::Stream as LocalStream;

            let dispatch = move |req| {
                let request_sender = request_sender.clone();
                async move {
                    request_sender
                        .send(req)
                        .await
                        .map_err(|_| "daemon channel closed".to_string())
                }
            };

            loop {
                let name = match ipc_name() {
                    Ok(name) => name,
                    Err(e) => {
                        tracing::error!("Failed to get IPC name in daemon: {e}");
                        continue;
                    }
                };
                if let Ok(stream) = LocalStream::connect(name) {
                    let mut reader = BufReader::new(stream);
                    let mut buf = String::new();
                    if let Err(e) = reader.read_line(&mut buf) {
                        tracing::error!("Failed to read line from IPC stream: {e}");
                        continue;
                    }
                    let req: Request = match serde_json::from_str(&buf) {
                        Ok(req) => req,
                        Err(e) => {
                            tracing::error!("Failed to parse request JSON: {e}");
                            continue;
                        }
                    };
                    handle_daemon_request(req, &dispatch).await;
                }
            }
        });
    });
}

pub fn nccl_daemon_replicator_mistralrs(mistralrs: Arc<crate::MistralRs>) {
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            use interprocess::local_socket::traits::Stream;
            use interprocess::local_socket::Stream as LocalStream;

            let dispatch = move |req| {
                let mistralrs = mistralrs.clone();
                async move {
                    mistralrs
                        .send_request_async(req)
                        .await
                        .map_err(|err| format!("{err:?}"))
                }
            };

            loop {
                let name = match ipc_name() {
                    Ok(name) => name,
                    Err(e) => {
                        tracing::error!("Failed to get IPC name in daemon: {e}");
                        continue;
                    }
                };
                if let Ok(stream) = LocalStream::connect(name) {
                    let mut reader = BufReader::new(stream);
                    let mut buf = String::new();
                    if let Err(e) = reader.read_line(&mut buf) {
                        tracing::error!("Failed to read line from IPC stream: {e}");
                        continue;
                    }
                    let req: Request = match serde_json::from_str(&buf) {
                        Ok(req) => req,
                        Err(e) => {
                            tracing::error!("Failed to parse request JSON: {e}");
                            continue;
                        }
                    };
                    handle_daemon_request(req, &dispatch).await;
                }
            }
        });
    });
}

pub fn ring_daemon_replicator(request_sender: Sender<Request>) {
    let ring_config = RingConfig::load();

    let master_ip = ring_config.master_ip();
    let master_port = ring_config.master_port;
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            let dispatch = move |req| {
                let request_sender = request_sender.clone();
                async move {
                    request_sender
                        .send(req)
                        .await
                        .map_err(|_| "daemon channel closed".to_string())
                }
            };

            loop {
                if let Ok(stream) = TcpStream::connect(format!("{master_ip}:{master_port}")) {
                    let mut reader = BufReader::new(stream);
                    let mut buf = String::new();
                    if let Err(e) = reader.read_line(&mut buf) {
                        tracing::error!("Failed to read line from ring stream: {e}");
                        continue;
                    }
                    let req: Request = match serde_json::from_str(&buf) {
                        Ok(req) => req,
                        Err(e) => {
                            tracing::error!("Failed to parse request JSON: {e}");
                            continue;
                        }
                    };
                    handle_daemon_request(req, &dispatch).await;
                }
            }
        });
    });
}

pub fn ring_daemon_replicator_mistralrs(mistralrs: Arc<crate::MistralRs>) {
    let ring_config = RingConfig::load();

    let master_ip = ring_config.master_ip();
    let master_port = ring_config.master_port;
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            let dispatch = move |req| {
                let mistralrs = mistralrs.clone();
                async move {
                    mistralrs
                        .send_request_async(req)
                        .await
                        .map_err(|err| format!("{err:?}"))
                }
            };

            loop {
                if let Ok(stream) = TcpStream::connect(format!("{master_ip}:{master_port}")) {
                    let mut reader = BufReader::new(stream);
                    let mut buf = String::new();
                    if let Err(e) = reader.read_line(&mut buf) {
                        tracing::error!("Failed to read line from ring stream: {e}");
                        continue;
                    }
                    let req: Request = match serde_json::from_str(&buf) {
                        Ok(req) => req,
                        Err(e) => {
                            tracing::error!("Failed to parse request JSON: {e}");
                            continue;
                        }
                    };
                    handle_daemon_request(req, &dispatch).await;
                }
            }
        });
    });
}

async fn handle_daemon_request<F, Fut>(req: Request, dispatch: &F)
where
    F: Fn(Request) -> Fut,
    Fut: Future<Output = Result<(), String>>,
{
    match req {
        Request::Detokenize(mut x) => {
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            x.response = sender;
            if let Err(e) = dispatch(Request::Detokenize(x)).await {
                tracing::error!("Daemon dispatch failed for Detokenize request: {e}");
                return;
            }
            match receiver.recv().await {
                Some(resp) => {
                    if let Err(e) = resp {
                        tracing::error!("Detokenize response error: {e}");
                    }
                }
                None => tracing::error!("Detokenize response channel closed"),
            }
        }
        Request::Tokenize(mut x) => {
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            x.response = sender;
            if let Err(e) = dispatch(Request::Tokenize(x)).await {
                tracing::error!("Daemon dispatch failed for Tokenize request: {e}");
                return;
            }
            match receiver.recv().await {
                Some(resp) => {
                    if let Err(e) = resp {
                        tracing::error!("Tokenize response error: {e}");
                    }
                }
                None => tracing::error!("Tokenize response channel closed"),
            }
        }
        Request::Normal(mut x) => {
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
            x.is_streaming = false;
            x.response = sender;
            if let Err(e) = dispatch(Request::Normal(x)).await {
                tracing::error!("Daemon dispatch failed for Normal request: {e}");
                return;
            }
            loop {
                match receiver.recv().await {
                    Some(crate::Response::AgenticToolCallProgress { .. })
                    | Some(crate::Response::BlockDenoisingProgress(_))
                    | Some(crate::Response::File(_)) => continue,
                    Some(resp) => {
                        if let Err(e) = resp.as_result() {
                            tracing::error!("Normal response error: {e}");
                        }
                        break;
                    }
                    None => {
                        tracing::error!("Normal response channel closed");
                        break;
                    }
                }
            }
        }
        req => {
            if let Err(e) = dispatch(req).await {
                tracing::error!("Daemon dispatch failed for request: {e}");
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub(crate) struct BigCCharArray(#[serde(with = "BigArray")] pub(crate) [c_char; 128]);

#[derive(Serialize, Deserialize, Debug)]
pub(crate) enum WorkerTransferData {
    Init {
        id: BigCCharArray,
        ids: Option<Vec<BigCCharArray>>,
        worker_rank: usize,
    },
}

pub fn begin_tensor_parallel_session(model_slots: usize) -> anyhow::Result<()> {
    if model_slots == 0 || is_daemon() || !mistralrs_quant::distributed::use_nccl() {
        return Ok(());
    }
    if TP_SESSION.get().is_some() {
        return Ok(());
    }

    let ids = (0..model_slots)
        .map(|_| mistralrs_quant::Id::new())
        .collect::<Vec<_>>();
    spawn_tensor_parallel_workers(&ids)?;
    let _ = TP_SESSION.set(Mutex::new(TensorParallelSession {
        ids,
        next_model_index: 0,
    }));
    Ok(())
}

fn spawn_tensor_parallel_workers(ids: &[mistralrs_quant::Id]) -> anyhow::Result<()> {
    let num_workers = mistralrs_quant::distributed::get_global_tp_size_from_devices()? - 1;
    if num_workers == 0 {
        return Ok(());
    }

    let payload_ids = ids.iter().map(id_to_payload).collect::<Vec<_>>();
    for worker_rank in 0..num_workers {
        let exe_path = env::current_exe().expect("Failed to get current exe");
        let args = env::args().collect::<Vec<_>>();
        let mut cmd = Command::new(exe_path);
        cmd.args(&args[1..]);
        let data = WorkerTransferData::Init {
            id: payload_ids[0].clone(),
            ids: Some(payload_ids.clone()),
            worker_rank,
        };

        cmd.env(IS_DAEMON_FLAG, serde_json::to_string(&data)?);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        cmd.stdin(std::process::Stdio::null());
        spawn_worker_and_reap(&mut cmd)?;
    }

    Ok(())
}

fn spawn_worker_and_reap(cmd: &mut Command) -> anyhow::Result<()> {
    let mut child = cmd.spawn().context("Failed to spawn process")?;
    std::thread::spawn(move || {
        if let Err(e) = child.wait() {
            tracing::error!("Failed to wait for worker process: {e}");
        }
    });
    Ok(())
}

fn id_to_payload(id: &mistralrs_quant::Id) -> BigCCharArray {
    BigCCharArray(*id.internal())
}

fn ensure_worker_session(ids: Option<&[BigCCharArray]>) {
    let Some(ids) = ids else {
        return;
    };
    if TP_SESSION.get().is_some() {
        return;
    }
    let ids = ids
        .iter()
        .map(|id| mistralrs_quant::Id::uninit(id.0))
        .collect::<Vec<_>>();
    let _ = TP_SESSION.set(Mutex::new(TensorParallelSession {
        ids,
        next_model_index: 0,
    }));
}

fn next_tensor_parallel_id() -> anyhow::Result<Option<mistralrs_quant::Id>> {
    let Some(session) = TP_SESSION.get() else {
        return Ok(None);
    };
    let mut session = session
        .lock()
        .map_err(|_| anyhow::anyhow!("Tensor parallel session lock poisoned"))?;
    let id = session
        .ids
        .get(session.next_model_index)
        .copied()
        .with_context(|| {
            format!(
                "Tensor parallel session has no communicator id for model index {}",
                session.next_model_index
            )
        })?;
    session.next_model_index += 1;
    Ok(Some(id))
}

fn send_worker_ready() -> anyhow::Result<()> {
    for _ in 0..WORKER_READY_CONNECT_ATTEMPTS {
        if let Ok(mut stream) = LocalStream::connect(ipc_name()?) {
            stream.write_all(b"ready\n")?;
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(WORKER_READY_CONNECT_SLEEP_MS));
    }

    let mut stream = LocalStream::connect(ipc_name()?)?;
    stream.write_all(b"ready\n")?;
    Ok(())
}

pub(crate) fn ipc_name() -> anyhow::Result<Name<'static>> {
    let printname = "mistralrs_daemon.sock";
    Ok(printname.to_ns_name::<GenericNamespaced>()?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_distributed_mapper<T: DeviceMappedModelLoader + IsqModelLoader + ?Sized>(
    dtype: DType,
    device: &Device,
    available_devices: &[Device],
    global_world_size_override: Option<usize>,
    silent: bool,
    config: &str,
    loading_isq: bool,
    from_uqff: bool,
    write_uqff: bool,
    organization: IsqOrganization,
    model: &T,
    paths: &dyn ModelPaths,
) -> anyhow::Result<(Box<dyn DeviceMapper + Send + Sync>, ShardedVarBuilder)> {
    if !(cfg!(feature = "cuda") || cfg!(feature = "ring")) {
        tracing::warn!(
            "Distributed support was not included in the build, be sure to build with `--features nccl`."
        );
    }

    // NCCL case!

    let local_world_size = available_devices.len();
    let global_world_size = if let Some(world_size) = global_world_size_override {
        world_size
    } else if let Ok(x) = std::env::var(MISTRALRS_MN_GLOBAL_WORLD_SIZE) {
        usize::from_str(&x).context("MISTRALRS_MN_GLOBAL_WORLD_SIZE")?
    } else {
        // global world size is always >= local world size
        std::cmp::max(
            mistralrs_quant::distributed::get_global_tp_size_from_devices()?,
            local_world_size,
        )
    };

    let use_multi_node = std::env::var(MISTRALRS_MN_GLOBAL_WORLD_SIZE).is_ok();
    if use_multi_node {
        info!("MISTRALRS_MN_GLOBAL_WORLD_SIZE is set, entering multi-node.");
    }

    if global_world_size < local_world_size || global_world_size % local_world_size != 0 {
        anyhow::bail!("Global world size {global_world_size} must both be at least and divide the local world size {local_world_size}");
    }

    // Sharded layers would serialize as rank-local slices.
    if write_uqff && global_world_size > 1 {
        anyhow::bail!(
            "Writing UQFF requires a single rank (got world size {global_world_size}); disable tensor parallelism."
        );
    }

    info!("Local tensor parallel world size is {local_world_size}");
    info!("Global tensor parallel world size is {global_world_size}");

    // TP uses parallel pipelines.
    let name = ipc_name()?;
    let mut id;
    let local_rank = if let Ok(payload) = env::var(IS_DAEMON_FLAG) {
        let payload: WorkerTransferData = serde_json::from_str(&payload)?;
        let WorkerTransferData::Init {
            id: new_id,
            ids,
            worker_rank,
        } = payload;
        ensure_worker_session(ids.as_deref());
        id = next_tensor_parallel_id()?.unwrap_or_else(|| mistralrs_quant::Id::uninit(new_id.0));

        send_worker_ready()?;
        worker_rank + 1
    } else if cfg!(feature = "ring") {
        id = mistralrs_quant::Id::new();

        let config = RingConfig::load();

        config.rank
    } else if let Some(session_id) = next_tensor_parallel_id()? {
        id = session_id;
        let num_workers = global_world_size - 1;
        let listener = ListenerOptions::new().name(name).create_sync()?;
        let mut ready_count = 0;

        while ready_count < num_workers {
            let stream = listener.accept()?;
            let mut reader = BufReader::new(stream);
            let mut message = String::new();
            reader.read_line(&mut message)?;
            if message.trim() == "ready" {
                ready_count += 1;
            }
        }
        info!("All workers have received the ids!");

        0
    } else {
        id = mistralrs_quant::Id::new();
        let num_workers = global_world_size - 1;
        let mut children = Vec::new();
        for worker_rank in 0..num_workers {
            let exe_path = env::current_exe().expect("Failed to get current exe");

            let args: Vec<String> = env::args().collect();

            let mut cmd = Command::new(exe_path);
            cmd.args(&args[1..]);

            let data = WorkerTransferData::Init {
                id: BigCCharArray(*id.internal()),
                ids: None,
                worker_rank,
            };

            cmd.env(IS_DAEMON_FLAG, serde_json::to_string(&data)?);

            cmd.stdout(std::process::Stdio::null());
            cmd.stderr(std::process::Stdio::null());
            cmd.stdin(std::process::Stdio::null());

            children.push(cmd.spawn().expect("Failed to spawn process"));
        }

        let listener = ListenerOptions::new().name(name).create_sync()?;
        let mut ready_count = 0;

        while ready_count < num_workers {
            let stream = listener.accept()?;
            let mut reader = BufReader::new(stream);
            let mut message = String::new();
            reader.read_line(&mut message)?;
            if message.trim() == "ready" {
                ready_count += 1;
            }
        }
        info!("All workers have received the ids!");

        0
    };

    if use_multi_node {
        if let Ok(n_nodes) = env::var("MISTRALRS_MN_HEAD_NUM_WORKERS") {
            let n_nodes = usize::from_str(&n_nodes).context("MISTRALRS_MN_HEAD_NUM_WORKERS")?;
            info!("Head node managing {n_nodes} workers.");
            let Ok(port) = env::var("MISTRALRS_MN_HEAD_PORT") else {
                anyhow::bail!("Got MISTRALRS_MN_HEAD_NUM_WORKERS, expected MISTRALRS_MN_HEAD_PORT");
            };
            info!("Head node initializing connection on {port}.");
            let server = mistralrs_quant::Server::new(
                &format!("0.0.0.0:{port}"),
                n_nodes,
                local_world_size,
            )?;

            server.broadcast_id(&id)?;
        } else if let Ok(addr) = env::var("MISTRALRS_MN_WORKER_SERVER_ADDR") {
            info!("Worker node connecting to {addr}.");
            let client = mistralrs_quant::Client::new(addr.parse()?, local_world_size)?;

            id = client.receive_id()?;
        }
    }

    let rank_offset = if env::var("MISTRALRS_MN_WORKER_SERVER_ADDR").is_ok() {
        let Ok(node_id) = env::var("MISTRALRS_MN_WORKER_ID") else {
            anyhow::bail!("Got MISTRALRS_MN_WORKER_SERVER_ADDR, expected MISTRALRS_MN_WORKER_ID");
        };
        let node_id = usize::from_str(&node_id).context("MISTRALRS_MN_WORKER_ID")?;
        info!("Worker ID is {node_id}.");
        (node_id + 1) * local_world_size
    } else {
        0
    };

    // They each block on each other
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html?ncclcomminitrank#ncclcomminitrank
    let comm = mistralrs_quant::Comm::from_device(
        id,
        device,
        local_rank + rank_offset,
        global_world_size,
    )?;

    let make_dummy_regexes = if loading_isq && from_uqff {
        // Dummy weights for the layers which will be overwritten...
        Some(std::sync::Arc::new(
            if matches!(organization, IsqOrganization::MoeExpertsOnly) {
                model.isq_layer_regexes_moqe(config)?
            } else {
                model.isq_layer_regexes(config)?
            },
        ))
    } else {
        None
    };

    let sharded_vb = varbuilder_utils::from_mmaped_safetensors(
        paths.get_weight_filenames().to_vec(),
        vec![],
        Some(dtype),
        &Device::Cpu,
        vec![],
        silent,
        make_dummy_regexes,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;

    info!("Loading all ranks.");
    // The mapper is specific to this pipeline
    let mapper = DeviceMapSetting::Nccl {
        nm_device: available_devices[0].clone(),
        comm: Arc::new(comm),
    }
    .into_mapper(model.num_layers(config)?, device, None, available_devices)?;

    let sharded_vb = if !loading_isq {
        sharded_vb.clone().set_device(device.clone())
    } else {
        sharded_vb.clone()
    };

    Ok((mapper, sharded_vb))
}

#[cfg(test)]
mod tests {
    use super::select_compatible_tensor_parallel_size;
    use crate::paged_attention::{KvCacheLayout, ModelConfigMetadata};

    fn config(num_attn_heads: usize, num_kv_heads: usize) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 1024,
            num_layers: 1,
            hidden_size: num_attn_heads * 64,
            num_kv_heads,
            num_attn_heads,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 64,
            kv_cache_layout: KvCacheLayout::Standard,
        }
    }

    #[test]
    fn auto_tp_keeps_compatible_size() {
        let cfg = config(12, 4);
        assert_eq!(
            select_compatible_tensor_parallel_size(&cfg, 4).unwrap(),
            Some(4)
        );
    }

    #[test]
    fn auto_tp_steps_down_to_compatible_size() {
        let cfg = config(9, 3);
        assert_eq!(
            select_compatible_tensor_parallel_size(&cfg, 4).unwrap(),
            Some(3)
        );
    }

    #[test]
    fn auto_tp_disables_when_no_compatible_distributed_size_exists() {
        let cfg = config(3, 1);
        assert_eq!(
            select_compatible_tensor_parallel_size(&cfg, 2).unwrap(),
            None
        );
    }
}
