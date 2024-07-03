use std::mem::size_of;

use bytemuck_derive::{Pod, Zeroable};

pub(crate) type TokenId = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct TokRxInfo {
    pub vocab_size: u32,
    pub tok_eos: TokenId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct U32Pair(pub u32, pub u32);

pub fn vec_from_bytes<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    if bytes.len() % size_of::<T>() != 0 {
        panic!(
            "vecT: got {} bytes, needed multiple of {}",
            bytes.len(),
            size_of::<T>()
        );
    }
    bytemuck::cast_slice(bytes).to_vec()
}

pub fn to_hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("")
}
