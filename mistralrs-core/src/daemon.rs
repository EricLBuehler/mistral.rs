use core::ffi::c_char;
use interprocess::local_socket::NameType;
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::process;
use tokio::runtime::Runtime;
use tokio::sync::mpsc::Sender;

use crate::Request;

pub(crate) const IS_DAEMON_FLAG: &str = "__MISTRALRS_DAEMON_INTERNAL";

pub fn is_daemon() -> bool {
    std::env::var(IS_DAEMON_FLAG).is_ok()
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
    assert!(GenericNamespaced::is_supported());
    let printname = "mistralrs_daemon.sock";
    Ok(printname.to_ns_name::<GenericNamespaced>()?)
}

pub(crate) fn launch_daemon(request_sender: Sender<Request>) {
    let rt = Runtime::new().unwrap();
    rt.block_on(async move {
        use interprocess::local_socket::traits::Stream;
        use interprocess::local_socket::Stream as LocalStream;

        let worker_rank = if let Ok(payload) = env::var(IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload).unwrap();
            let WorkerTransferData::Init { worker_rank, .. } = payload;
            (worker_rank + 1) as u32
        } else {
            unreachable!()
        };

        let mut last_request = None;
        loop {
            let name = ipc_name().unwrap();

            match LocalStream::connect(name) {
                Ok(mut stream) => {
                    let req: Request = {
                        let mut reader = BufReader::new(&stream);
                        let mut buf = String::new();
                        let res = reader.read_line(&mut buf);
                        if res.is_err() {
                            break;
                        }

                        if last_request.as_ref().is_some_and(|last| last == &buf) || buf.is_empty()
                        {
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            continue;
                        } else {
                            last_request = Some(buf.clone());
                        }

                        serde_json::from_str(&buf).expect(&buf)
                    };
                    {
                        let x: [u8; 4] = worker_rank.to_le_bytes();
                        stream.write_all(&x).unwrap();
                    }

                    let request_sender = request_sender.clone();
                    tokio::spawn(async move {
                        match req {
                            Request::ActivateAdapters(x) => request_sender
                                .send(Request::ActivateAdapters(x))
                                .await
                                .unwrap(),
                            Request::ReIsq(x) => {
                                request_sender.send(Request::ReIsq(x)).await.unwrap()
                            }
                            Request::Terminate => {
                                request_sender.send(Request::Terminate).await.unwrap()
                            }
                            Request::Detokenize(mut x) => {
                                let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                                x.response = sender;
                                let req = Request::Detokenize(x);

                                request_sender.send(req).await.unwrap();
                                let resp = receiver.recv().await.unwrap();
                                resp.unwrap();
                            }
                            Request::Tokenize(mut x) => {
                                let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                                x.response = sender;
                                let req = Request::Tokenize(x);

                                request_sender.send(req).await.unwrap();
                                let resp = receiver.recv().await.unwrap();
                                resp.unwrap();
                            }
                            Request::Normal(mut x) => {
                                let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
                                x.is_streaming = false;
                                x.response = sender;
                                let req = Request::Normal(x);

                                request_sender.send(req).await.unwrap();
                                let resp = receiver.recv().await.unwrap();
                                resp.as_result().unwrap();
                            }
                            Request::TerminateAllSeqsNextStep => request_sender
                                .send(Request::TerminateAllSeqsNextStep)
                                .await
                                .unwrap(),
                        }
                    });
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => {
                    // panic!("{e:?}");
                }
            }
        }
    });
}
