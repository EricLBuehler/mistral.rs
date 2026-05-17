//! `setrlimit` wrappers.

use std::io;

use nix::sys::resource::{getrlimit, setrlimit, Resource};

const PROC_DIR: &str = "/proc";
const PROC_STATUS: &str = "status";
const STATUS_UID_PREFIX: &str = "Uid:";
const STATUS_THREADS_PREFIX: &str = "Threads:";

#[inline]
pub(crate) fn set(resource: Resource, value: u64) -> io::Result<()> {
    let (_, hard) = getrlimit(resource).map_err(|e| io::Error::from_raw_os_error(e as i32))?;
    let value = value.min(hard);
    setrlimit(resource, value, value).map_err(|e| io::Error::from_raw_os_error(e as i32))
}

pub(crate) fn nproc_limit(headroom: u64) -> u64 {
    current_uid_task_count()
        .map(|count| count.saturating_add(headroom))
        .unwrap_or(headroom)
}

#[inline]
pub(crate) fn zero(resource: Resource) -> io::Result<()> {
    set(resource, 0)
}

fn current_uid_task_count() -> Option<u64> {
    let uid = unsafe { libc::getuid() };
    let entries = std::fs::read_dir(PROC_DIR).ok()?;
    let mut total = 0u64;

    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_str().and_then(|s| s.parse::<u32>().ok()).is_none() {
            continue;
        }
        let Ok(status) = std::fs::read_to_string(entry.path().join(PROC_STATUS)) else {
            continue;
        };
        if status_uid(&status) == Some(uid) {
            total = total.saturating_add(status_threads(&status).unwrap_or(1));
        }
    }

    Some(total)
}

fn status_uid(status: &str) -> Option<u32> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(STATUS_UID_PREFIX))
        .and_then(|line| line.split_whitespace().next())
        .and_then(|uid| uid.parse().ok())
}

fn status_threads(status: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(STATUS_THREADS_PREFIX))
        .and_then(|line| line.split_whitespace().next())
        .and_then(|threads| threads.parse().ok())
}
