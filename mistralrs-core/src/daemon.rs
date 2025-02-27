use core::ffi::c_char;
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

pub(crate) const IS_DAEMON_FLAG: &str = "__MISTRALRS_DAEMON_INTERNAL";

pub(crate) fn is_daemon() -> bool {
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
    let printname = "mistralrs_daemon.sock";
    Ok(printname.to_ns_name::<GenericNamespaced>()?)
}
