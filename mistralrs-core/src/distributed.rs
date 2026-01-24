use anyhow::Context;
use candle_core::{DType, Device};
use core::ffi::c_char;
use interprocess::local_socket::traits::{Listener, Stream};
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
pub use mistralrs_quant::distributed::use_nccl;
use mistralrs_quant::{RingConfig, ShardedVarBuilder};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::Command;
use std::str::FromStr;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::Sender;
use tracing::info;

use crate::device_map::DeviceMapper;
use crate::pipeline::{DeviceMappedModelLoader, IsqModelLoader};
use crate::utils::varbuilder_utils::{self, DeviceForLoadTensor};
use crate::{DeviceMapSetting, IsqOrganization, ModelPaths, Request};

pub(crate) const IS_DAEMON_FLAG: &str = "__MISTRALRS_DAEMON_INTERNAL";

pub fn is_daemon() -> bool {
    if cfg!(feature = "cuda") && !cfg!(feature = "ring") {
        std::env::var(IS_DAEMON_FLAG).is_ok()
    } else if cfg!(feature = "ring") {
        !RingConfig::load().is_master_rank()
    } else {
        false
    }
}

pub fn nccl_daemon_replicator(request_sender: Sender<Request>) {
    use std::io::BufRead;
    use std::io::BufReader;

    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            use interprocess::local_socket::traits::Stream;
            use interprocess::local_socket::Stream as LocalStream;

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
                    let mut req: Request = match serde_json::from_str(&buf) {
                        Ok(req) => req,
                        Err(e) => {
                            tracing::error!("Failed to parse request JSON: {e}");
                            continue;
                        }
                    };

                    req = match req {
                        Request::ReIsq(x) => Request::ReIsq(x),
                        Request::Terminate => Request::Terminate,
                        Request::Detokenize(mut x) => {
                            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                            x.response = sender;
                            let req = Request::Detokenize(x);

                            if request_sender.send(req).await.is_err() {
                                tracing::error!("Daemon channel closed for Detokenize request");
                                continue;
                            }
                            match receiver.recv().await {
                                Some(resp) => {
                                    if let Err(e) = resp {
                                        tracing::error!("Detokenize response error: {e}");
                                    }
                                }
                                None => tracing::error!("Detokenize response channel closed"),
                            }
                            continue;
                        }
                        Request::Tokenize(mut x) => {
                            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                            x.response = sender;
                            let req = Request::Tokenize(x);

                            if request_sender.send(req).await.is_err() {
                                tracing::error!("Daemon channel closed for Tokenize request");
                                continue;
                            }
                            match receiver.recv().await {
                                Some(resp) => {
                                    if let Err(e) = resp {
                                        tracing::error!("Tokenize response error: {e}");
                                    }
                                }
                                None => tracing::error!("Tokenize response channel closed"),
                            }
                            continue;
                        }
                        Request::Normal(mut x) => {
                            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                            x.is_streaming = false;
                            x.response = sender;
                            let req = Request::Normal(x);

                            if request_sender.send(req).await.is_err() {
                                tracing::error!("Daemon channel closed for Normal request");
                                continue;
                            }
                            match receiver.recv().await {
                                Some(resp) => {
                                    if let Err(e) = resp.as_result() {
                                        tracing::error!("Normal response error: {e}");
                                    }
                                }
                                None => tracing::error!("Normal response channel closed"),
                            }
                            continue;
                        }
                        Request::TerminateAllSeqsNextStep => Request::TerminateAllSeqsNextStep,
                    };

                    if request_sender.send(req).await.is_err() {
                        tracing::error!("Daemon channel closed for request");
                    }
                }
            }
        });
    });
}

pub fn ring_daemon_replicator(request_sender: Sender<Request>) {
    use std::io::BufRead;
    use std::io::BufReader;

    let ring_config = RingConfig::load();

    let master_ip = ring_config.master_ip();
    let master_port = ring_config.master_port;
    std::thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async move {
            loop {
                if let Ok(stream) = TcpStream::connect(format!("{master_ip}:{master_port}")) {
                    let mut reader = BufReader::new(stream);
                    let mut buf = String::new();
                    reader.read_line(&mut buf).unwrap();
                    let mut req: Request = serde_json::from_str(&buf).unwrap();

                    req = match req {
                        Request::ReIsq(x) => Request::ReIsq(x),
                        Request::Terminate => Request::Terminate,
                        Request::Detokenize(mut x) => {
                            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                            x.response = sender;
                            let req = Request::Detokenize(x);

                            request_sender.send(req).await.unwrap();
                            let resp = receiver.recv().await.unwrap();
                            resp.unwrap();
                            continue;
                        }
                        Request::Tokenize(mut x) => {
                            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                            x.response = sender;
                            let req = Request::Tokenize(x);

                            request_sender.send(req).await.unwrap();
                            let resp = receiver.recv().await.unwrap();
                            resp.unwrap();
                            continue;
                        }
                        Request::Normal(mut x) => {
                            let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                            x.is_streaming = false;
                            x.response = sender;
                            let req = Request::Normal(x);

                            request_sender.send(req).await.unwrap();
                            let resp = receiver.recv().await.unwrap();
                            resp.as_result().unwrap();
                            continue;
                        }
                        Request::TerminateAllSeqsNextStep => Request::TerminateAllSeqsNextStep,
                    };

                    request_sender.send(req).await.unwrap();
                }
            }
        });
    });
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub(crate) struct BigCCharArray(#[serde(with = "BigArray")] pub(crate) [c_char; 128]);

#[derive(Serialize, Deserialize, Debug)]
pub(crate) enum WorkerTransferData {
    Init {
        id: BigCCharArray,
        worker_rank: usize,
    },
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
    silent: bool,
    config: &str,
    loading_isq: bool,
    from_uqff: bool,
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
    let global_world_size = if let Ok(x) = std::env::var("MISTRALRS_MN_GLOBAL_WORLD_SIZE") {
        usize::from_str(&x).context("MISTRALRS_MN_GLOBAL_WORLD_SIZE")?
    } else {
        // global world size is always >= local world size
        std::cmp::max(
            mistralrs_quant::distributed::get_global_tp_size_from_devices()?,
            local_world_size,
        )
    };

    let use_multi_node = std::env::var("MISTRALRS_MN_GLOBAL_WORLD_SIZE").is_ok();
    if use_multi_node {
        info!("MISTRALRS_MN_GLOBAL_WORLD_SIZE is set, entering multi-node.");
    }

    if global_world_size < local_world_size || global_world_size % local_world_size != 0 {
        anyhow::bail!("Global world size {global_world_size} must both be at least and divide the local world size {local_world_size}");
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
            worker_rank,
        } = payload;
        id = mistralrs_quant::Id::uninit(new_id.0);

        let mut stream = LocalStream::connect(name)?;
        stream.write_all(b"ready\n")?;
        worker_rank + 1
    } else if cfg!(feature = "ring") {
        id = mistralrs_quant::Id::new();

        let config = RingConfig::load();

        config.rank
    } else {
        id = mistralrs_quant::Id::new();
        let num_ranks = mistralrs_quant::distributed::get_global_tp_size_from_devices()?;
        let num_workers = num_ranks - 1;
        let mut children = Vec::new();
        for worker_rank in 0..num_workers {
            let exe_path = env::current_exe().expect("Failed to get current exe");

            let args: Vec<String> = env::args().collect();

            let mut cmd = Command::new(exe_path);
            cmd.args(&args[1..]);

            let data = WorkerTransferData::Init {
                id: BigCCharArray(*id.internal()),
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
