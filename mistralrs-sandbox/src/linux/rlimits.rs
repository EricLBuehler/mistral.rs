//! `setrlimit` wrappers.

use std::io;

use nix::sys::resource::{setrlimit, Resource};

#[inline]
pub(crate) fn set(resource: Resource, value: u64) -> io::Result<()> {
    setrlimit(resource, value, value).map_err(|e| io::Error::from_raw_os_error(e as i32))
}

#[inline]
pub(crate) fn zero(resource: Resource) -> io::Result<()> {
    set(resource, 0)
}
