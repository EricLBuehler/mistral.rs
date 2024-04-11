use std::{fmt::Debug, ops::Index};

use super::bytes::TokenId;

#[derive(Clone)]
pub struct SimpleVob {
    data: Vec<u32>,
}

impl Debug for SimpleVob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleVob")
            .field("len", &self.len())
            .finish()
    }
}

impl Default for SimpleVob {
    fn default() -> Self {
        Self::new()
    }
}

const BITS: usize = 32;

impl SimpleVob {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn alloc(size: usize) -> Self {
        let mut r = Self::new();
        r.resize(size);
        r
    }

    pub fn len(&self) -> usize {
        self.data.len() * BITS
    }

    pub fn num_set(&self) -> usize {
        self.data.iter().map(|x| x.count_ones() as usize).sum()
    }
    #[allow(clippy::cast_possible_truncation)]
    pub fn negated(&self, size: usize) -> Self {
        let mut r = Self::new();
        r.data = self.data.iter().map(|x| !x).collect();
        for i in size..r.len() {
            // disallow tokens that are out of range
            r.disallow_token(i as TokenId);
        }
        r
    }

    pub unsafe fn as_ptr(&self) -> *const u32 {
        self.data.as_ptr()
    }

    #[inline(always)]
    pub fn allow_token(&mut self, tok: TokenId) {
        let idx = tok as usize;
        let byte_idx = idx / BITS;
        let bit_idx = idx % BITS;
        self.data[byte_idx] |= 1 << bit_idx;
    }

    #[inline(always)]
    pub fn disallow_token(&mut self, tok: TokenId) {
        let idx = tok as usize;
        let byte_idx = idx / BITS;
        let bit_idx = idx % BITS;
        self.data[byte_idx] &= !(1 << bit_idx);
    }

    pub fn set(&mut self, tok: TokenId, val: bool) {
        if val {
            self.allow_token(tok);
        } else {
            self.disallow_token(tok);
        }
    }

    pub fn resize(&mut self, size: usize) {
        let new_size = size / BITS + 1;
        assert!(new_size >= self.data.len());
        self.data.resize(new_size, 0);
    }

    #[inline(always)]
    pub fn is_allowed(&self, tok: TokenId) -> bool {
        let idx = tok as usize;
        let byte_idx = idx / 32;
        let bit_idx = idx % 32;
        (self.data[byte_idx] & (1 << bit_idx)) != 0
    }

    pub fn set_all(&mut self, val: bool) {
        let val = if val { !0 } else { 0 };
        self.data.iter_mut().for_each(|x| *x = val);
    }

    pub fn apply_to(&self, logits: &mut [f32]) {
        for (idx, v) in self.data.iter().enumerate() {
            if *v == 0 {
                continue;
            }
            let idx = idx * BITS;
            for bit_idx in 0..BITS {
                if v & (1 << bit_idx) != 0 {
                    logits[idx + bit_idx] = 0.0;
                }
            }
        }
    }
}

impl Index<usize> for SimpleVob {
    type Output = bool;
    #[allow(clippy::cast_possible_truncation)]
    fn index(&self, index: usize) -> &Self::Output {
        if self.is_allowed(index as TokenId) {
            &true
        } else {
            &false
        }
    }
}
