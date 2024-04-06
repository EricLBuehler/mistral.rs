use aici_abi::{
    bytes::{limit_bytes, limit_str},
    toktree::TokTrie,
};
use aicirt::{
    api::{
        AiciMidProcessReq, AiciMidProcessResp, AiciPostPreProcessReq, AiciPostPreProcessResp,
        AiciPreProcessResultInner, AuthInfo, GetTagsResp, InstantiateReq, MkModuleReq,
        MkModuleResp, SequenceResult, SetTagsReq, TokensResp,
    },
    futexshm::ClientChannel,
    msgchannel::MessageChannel,
    shm::{Shm, Unlink},
    user_error, HashMap,
};
use anyhow::Result;
use futures::future::select_all;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    process::{Child, Command},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};
use tokio::{signal::unix::SignalKind, sync::oneshot};

pub struct CmdChannel {
    cmd_pending: bool,
    cmd_ch: ClientChannel,
    resp_ch: ClientChannel,
    #[allow(dead_code)]
    busy_wait_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Empty {}

const M: usize = 1 << 20;

fn build_ch(name: &str, json_size: usize) -> Result<ClientChannel> {
    let shm = Shm::new(name, json_size * M, Unlink::Pre)?;
    Ok(ClientChannel::new(shm))
}

impl CmdChannel {
    pub fn new(
        json_size: usize,
        pref: &str,
        suff: &str,
        busy_wait_duration: Duration,
    ) -> Result<Self> {
        Ok(Self {
            cmd_pending: false,
            cmd_ch: build_ch(&format!("{}cmd{}", pref, suff), json_size)?,
            resp_ch: build_ch(&format!("{}resp{}", pref, suff), json_size)?,
            busy_wait_duration,
        })
    }

    pub fn send_bytes(&mut self, data: &[u8]) -> Result<()> {
        assert!(!self.cmd_pending);
        self.cmd_pending = true;
        self.cmd_ch.send_req(data)?;
        Ok(())
    }

    pub fn exec<T: Serialize, R>(&mut self, op: &str, data: T) -> Result<R>
    where
        R: for<'d> Deserialize<'d>,
    {
        self.send(op, data)?;
        self.expect(&format!("cmd:{}", op))
    }

    pub fn send<T: Serialize>(&mut self, op: &str, data: T) -> Result<()> {
        let mut value = serde_json::to_value(data)?;
        value["op"] = json!(op);
        let bytes = serde_json::to_vec(&value)?;
        self.send_bytes(&bytes)
    }

    pub fn expect<R>(&mut self, ctx: &str) -> Result<R>
    where
        R: for<'d> Deserialize<'d>,
    {
        assert!(self.cmd_pending);
        let bytes = self.resp_ch.recv_resp(Duration::MAX).unwrap();
        self.cmd_pending = false;
        let mut resp: Value = serde_json::from_slice(&bytes)?;
        if resp["type"] != "ok" {
            return Err(anyhow::anyhow!(
                "Bad response ({ctx}): {}",
                limit_bytes(&bytes, 500)
            ));
        }
        let data = resp
            .as_object_mut()
            .unwrap()
            .remove("data")
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Bad response ({ctx}) - no 'data': {}",
                    limit_bytes(&bytes, 500)
                )
            })?;
        let resp = serde_json::from_value(data).map_err(|e| {
            anyhow::anyhow!("Bad response ({ctx}): {e} {}", limit_bytes(&bytes, 500))
        })?;
        Ok(resp)
    }
}

pub struct AiciRtIface {
    cmd: CmdChannel,
    pub pending_mid_size: usize,
    pub bin_shm: Shm,
    pub side_cmd: AsyncCmdChannel,
    #[allow(dead_code)]
    child: Child,
}

pub struct Args {
    pub aicirt: String,
    pub tokenizer: String,
    pub json_size: usize,
    pub bin_size: usize,
    pub shm_prefix: String,
    pub busy_wait_time: u64,
    pub add_args: Vec<String>,
}
#[allow(dead_code)] // TODO: remove
pub fn kill_self() {
    unsafe {
        libc::kill(libc::getpid(), libc::SIGTERM);
    }
}
#[allow(dead_code)] // TODO: remove
impl AiciRtIface {
    pub fn start_aicirt(args: &Args, tok_trie: &TokTrie) -> Result<Self> {
        let busy_wait_time = Duration::from_millis(args.busy_wait_time);
        let shm_name = MessageChannel::shm_name(&(args.shm_prefix.clone() + "bin"));
        let cmd = CmdChannel::new(args.json_size, &args.shm_prefix, "", busy_wait_time)?;
        let side_cmd = AsyncCmdChannel::new(args.json_size, &args.shm_prefix, "-side")?;
        let bin_shm = Shm::new(&shm_name, args.bin_size * M, Unlink::Pre)?;

        let mut cmd_bld = Command::new(&args.aicirt);
        cmd_bld
            .arg("--tokenizer")
            .arg(&args.tokenizer)
            .arg("--json-size")
            .arg(&args.json_size.to_string())
            .arg("--bin-size")
            .arg(&args.bin_size.to_string())
            .arg("--name")
            .arg(&args.shm_prefix)
            .arg("--futex");
        for a in &args.add_args {
            cmd_bld.arg(a);
        }
        let child = cmd_bld.arg("--server").spawn()?;

        let pid = child.id() as libc::c_int;
        let default_panic_hook = std::panic::take_hook();

        std::panic::set_hook(Box::new(move |panic_info| {
            eprintln!("killing {pid}");
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
            default_panic_hook(panic_info);
            std::process::exit(100);
        }));

        let _killer = tokio::spawn(async move {
            let sigs = vec![
                SignalKind::interrupt(),
                SignalKind::quit(),
                SignalKind::terminate(),
            ];

            let mut sigs = sigs
                .iter()
                .map(|s| tokio::signal::unix::signal(*s).unwrap())
                .collect::<Vec<_>>();

            loop {
                let futures: Vec<_> = sigs.iter_mut().map(|s| s.recv()).collect();
                let pinned_futures: Vec<_> = futures.into_iter().map(|f| Box::pin(f)).collect();
                select_all(pinned_futures).await;
                // log::info!("Killing child process");
                unsafe {
                    libc::kill(-pid, libc::SIGTERM);
                }
            }
        });

        let mut r = Self {
            cmd,
            side_cmd,
            bin_shm,
            child,
            pending_mid_size: usize::MAX,
        };

        let _: Value = r.cmd.exec("ping", json!({}))?;
        let tokens: TokensResp = r
            .cmd
            .exec("tokens", json!({}))
            .map_err(|e| anyhow::anyhow!("check for pending aicirt processes! {e}"))?;

        // well, this is somewhat unlikely as we're passing the same tokenizer name down...
        if tokens.vocab_size != tok_trie.info().vocab_size {
            return Err(anyhow::anyhow!(
                "Vocab size mismatch: {:?} != {:?}",
                tokens,
                tok_trie.info()
            ));
        }

        Ok(r)
    }

    pub fn start_mid_process(&mut self, req: AiciMidProcessReq) -> Result<()> {
        assert!(self.pending_mid_size == usize::MAX);
        self.pending_mid_size = req.ops.len();
        self.cmd.send("mid_process", req)
    }

    pub fn finish_mid_process(&mut self) -> Result<AiciMidProcessResp> {
        assert!(self.pending_mid_size < usize::MAX);
        let r: AiciMidProcessResp = self.cmd.expect("async:mid_process")?;
        assert!(r.num_seqs == self.pending_mid_size);
        self.pending_mid_size = usize::MAX;
        Ok(r)
    }

    pub fn post_pre_process(
        &mut self,
        req: AiciPostPreProcessReq,
    ) -> Result<AiciPostPreProcessResp> {
        self.cmd.exec("post_pre_process", req)
    }
}

#[derive(Clone)]
pub struct AsyncCmdChannel {
    pending_reqs: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    cmd_ch: Arc<Mutex<ClientChannel>>,
}
#[allow(dead_code)] // TODO: remove
impl AsyncCmdChannel {
    pub fn new(json_size: usize, pref: &str, suff: &str) -> Result<Self> {
        let cmd = CmdChannel::new(json_size, pref, suff, Duration::ZERO)?;
        let pending_reqs = Arc::new(Mutex::new(
            HashMap::<String, oneshot::Sender<Value>>::default(),
        ));
        {
            let mut resp_ch = cmd.resp_ch;
            let pending_reqs = pending_reqs.clone();
            thread::spawn(move || loop {
                let resp = resp_ch.recv_resp2(Duration::ZERO, Duration::MAX).unwrap();
                let resp: Value = serde_json::from_slice(&resp).unwrap();
                let rid = resp["$rid"].as_str().unwrap().to_string();
                let tx = pending_reqs.lock().unwrap().remove(&rid).unwrap();
                tx.send(resp).unwrap();
            });
        }

        Ok(Self {
            pending_reqs,
            cmd_ch: Arc::new(Mutex::new(cmd.cmd_ch)),
        })
    }

    pub async fn set_tags(&self, req: SetTagsReq, authinfo: AuthInfo) -> Result<GetTagsResp> {
        self.exec("set_tags", req, authinfo).await
    }

    pub async fn get_tags(&self, authinfo: AuthInfo) -> Result<GetTagsResp> {
        self.exec("get_tags", json!({}), authinfo).await
    }

    pub async fn mk_module(&self, req: MkModuleReq, authinfo: AuthInfo) -> Result<MkModuleResp> {
        self.exec("mk_module", req, authinfo).await
    }

    pub async fn instantiate(
        &self,
        req: InstantiateReq,
        authinfo: AuthInfo,
    ) -> Result<SequenceResult<AiciPreProcessResultInner>> {
        self.exec("instantiate", req, authinfo).await
    }

    pub async fn exec<T: Serialize, R>(&self, op: &str, data: T, authinfo: AuthInfo) -> Result<R>
    where
        R: for<'d> Deserialize<'d>,
    {
        let rid = uuid::Uuid::new_v4().to_string();
        let mut data = serde_json::to_value(data)?;
        data["op"] = Value::String(op.to_string());
        data["$rid"] = Value::String(rid.clone());
        data["$auth"] = serde_json::to_value(authinfo)?;

        let (tx, rx) = oneshot::channel();
        self.pending_reqs.lock().unwrap().insert(rid.clone(), tx);

        self.cmd_ch
            .lock()
            .unwrap()
            .send_req(&serde_json::to_vec(&data)?)?;

        let mut resp = rx.await?;

        match resp["type"].as_str() {
            Some("ok") => {
                let data = resp.as_object_mut().unwrap().remove("data");
                if data.is_none() {
                    anyhow::bail!(
                        "Bad response ({op}) - no 'data': {}",
                        limit_bytes(&serde_json::to_vec(&resp)?, 500)
                    );
                }
                let data = data.unwrap();
                let data_copy = limit_bytes(&serde_json::to_vec(&data).unwrap(), 500);
                let resp = serde_json::from_value(data)
                    .map_err(|e| anyhow::anyhow!("Bad response ({op}): {e} {}", data_copy))?;
                Ok(resp)
            }
            _ => {
                let info = match resp["error"].as_str() {
                    Some(text) => text.to_string(),
                    _ => serde_json::to_string(&resp)?,
                };
                if resp["is_user_error"].as_bool().unwrap_or(false) {
                    Err(user_error!("While executing {op}:\n{info}"))
                } else {
                    Err(anyhow::anyhow!(
                        "Bad response ({op}): {}",
                        limit_str(&info, 2000)
                    ))
                }
            }
        }
    }
}
