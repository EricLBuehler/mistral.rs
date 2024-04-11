use std::{mem::size_of, slice::from_raw_parts};

pub(crate) type TokenId = u32;

#[repr(C)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TokRxInfo {
    pub vocab_size: u32,
    pub tok_eos: TokenId,
}

pub fn clone_vec_as_bytes<T>(input: &[T]) -> Vec<u8> {
    unsafe {
        let byte_slice = from_raw_parts(input.as_ptr() as *const u8, input.len() * size_of::<T>());
        byte_slice.to_vec()
    }
}

pub fn clone_as_bytes<T>(input: &T) -> Vec<u8> {
    unsafe {
        let byte_slice = from_raw_parts(input as *const T as *const u8, size_of::<T>());
        byte_slice.to_vec()
    }
}

pub fn box_from_bytes<T>(bytes: &[u8]) -> Box<T> {
    if bytes.len() != size_of::<T>() {
        panic!("T: got {} bytes, needed {}", bytes.len(), size_of::<T>());
    }
    let mut t: Box<T> = Box::new(unsafe { std::mem::zeroed() });
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), &mut *t as *mut T as *mut u8, size_of::<T>());
    }
    t
}

pub fn vec_from_bytes<T>(bytes: &[u8]) -> Vec<T> {
    if bytes.len() % size_of::<T>() != 0 {
        panic!(
            "vecT: got {} bytes, needed multiple of {}",
            bytes.len(),
            size_of::<T>()
        );
    }
    let num_elements = bytes.len() / size_of::<T>();
    let mut result = Vec::with_capacity(num_elements);
    unsafe {
        result.set_len(num_elements);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.as_mut_ptr() as *mut u8, bytes.len());
    }
    result
}

pub fn to_hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("")
}
