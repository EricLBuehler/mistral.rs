use std::{fmt::Debug, hash::Hash, ops::Index};

pub type TokenId = u32;

#[derive(Clone)]
pub struct SimpleVob {
    data: Vec<u32>,
    size: usize,
}

impl Hash for SimpleVob {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.size.hash(state);
        self.data.hash(state);
    }
}

impl PartialEq for SimpleVob {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.data == other.data
    }
}

impl Eq for SimpleVob {}

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

#[allow(clippy::from_over_into)]
impl Into<Vec<u32>> for SimpleVob {
    fn into(self) -> Vec<u32> {
        self.data
    }
}

const BITS: usize = 32;

impl SimpleVob {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            size: 0,
        }
    }

    pub fn from_slice(bits: &[bool]) -> Self {
        let mut r = Self::alloc(bits.len());
        for (idx, b) in bits.iter().enumerate() {
            r.set(idx, *b);
        }
        r
    }

    pub fn alloc(size: usize) -> Self {
        let mut r = Self::new();
        r.resize(size);
        r
    }

    pub fn all_true(size: usize) -> Self {
        let mut r = Self::alloc(size);
        r.set_all(true);
        r
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn num_set(&self) -> usize {
        self.data.iter().map(|x| x.count_ones() as usize).sum()
    }

    fn clear_excessive_bits(&mut self) {
        for i in self.size..(self.data.len() * 32) {
            // disallow tokens that are out of range
            self.disallow_token(i as TokenId);
        }
    }

    pub fn to_bin_string(&self) -> String {
        let mut s = String::new();
        for i in 0..self.size {
            s.push(if self.is_allowed(i as TokenId) {
                '1'
            } else {
                '0'
            });
        }
        s
    }

    pub fn negated(&self) -> Self {
        let mut r = Self::new();
        r.data = self.data.iter().map(|x| !x).collect();
        r.size = self.size;
        r.clear_excessive_bits();
        r
    }

    pub unsafe fn as_ptr(&self) -> *const u32 {
        self.data.as_ptr()
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    #[inline(always)]
    pub fn iter_set_entries(&self, mut f: impl FnMut(usize)) {
        let src = self.as_slice();
        let numelts = self.size;
        let max_len = numelts / 32;
        for (idx, d) in src.iter().enumerate().take(max_len) {
            // optimize for the two common cases
            if *d == 0 {
                continue;
            } else if *d == u32::MAX {
                for bit in 0..32 {
                    f(idx * 32 + bit);
                }
            } else {
                for bit in 0..32 {
                    if d & (1 << bit) != 0 {
                        f(idx * 32 + bit);
                    }
                }
            }
        }
        // final few elts
        for idx in (max_len * 32)..numelts {
            if self.is_allowed(idx as TokenId) {
                f(idx);
            }
        }
    }

    #[inline(always)]
    pub fn iter_unset_entries(&self, mut f: impl FnMut(usize)) {
        let src = self.as_slice();
        let numelts = self.size;
        let max_len = numelts / 32;
        for (idx, d) in src.iter().enumerate().take(max_len) {
            // optimize for the two common cases
            if *d == 0 {
                for bit in 0..32 {
                    f(idx * 32 + bit);
                }
            } else if *d == u32::MAX {
                continue;
            } else {
                for bit in 0..32 {
                    if d & (1 << bit) == 0 {
                        f(idx * 32 + bit);
                    }
                }
            }
        }
        // final few elts
        for idx in (max_len * 32)..numelts {
            if !self.is_allowed(idx as TokenId) {
                f(idx);
            }
        }
    }

    #[inline(always)]
    pub fn iter_entries(&self, mut f: impl FnMut(bool, usize)) {
        let src = self.as_slice();
        let numelts = self.size;
        let max_len = numelts / 32;
        for (idx, d) in src.iter().enumerate().take(max_len) {
            // optimize for the two common cases
            if *d == 0 {
                for bit in 0..32 {
                    f(false, idx * 32 + bit);
                }
            } else if *d == u32::MAX {
                for bit in 0..32 {
                    f(true, idx * 32 + bit);
                }
            } else {
                for bit in 0..32 {
                    f(d & (1 << bit) != 0, idx * 32 + bit);
                }
            }
        }
        // final few elts
        for idx in (max_len * 32)..numelts {
            f(self.is_allowed(idx as TokenId), idx);
        }
    }

    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() == self.data.len() * 4);
        bytemuck::cast_slice_mut(buf).copy_from_slice(&self.data);
    }

    #[inline(always)]
    pub fn allow_token(&mut self, tok: TokenId) {
        self.set(tok as usize, true)
    }

    #[inline(always)]
    pub fn disallow_token(&mut self, tok: TokenId) {
        self.set(tok as usize, false)
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, val: bool) {
        let byte_idx = idx / BITS;
        let bit_idx = idx % BITS;
        if val {
            self.data[byte_idx] |= 1 << bit_idx;
        } else {
            self.data[byte_idx] &= !(1 << bit_idx);
        }
    }

    pub fn resize(&mut self, size: usize) {
        let new_size = size / BITS + 1;
        assert!(new_size >= self.data.len());
        self.data.resize(new_size, 0);
        self.size = size;
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> bool {
        let byte_idx = idx / 32;
        let bit_idx = idx % 32;
        (self.data[byte_idx] & (1 << bit_idx)) != 0
    }

    #[inline(always)]
    pub fn is_allowed(&self, tok: TokenId) -> bool {
        self.get(tok as usize)
    }

    pub fn set_all(&mut self, val: bool) {
        let val = if val { !0 } else { 0 };
        self.data.iter_mut().for_each(|x| *x = val);
        self.clear_excessive_bits();
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

    pub fn iter(&self) -> SimpleVobIter {
        SimpleVobIter { vob: self, idx: 0 }
    }

    pub fn or(&mut self, other: &SimpleVob) {
        assert_eq!(self.size, other.size);
        for (idx, v) in self.data.iter_mut().zip(other.data.iter()) {
            *idx |= *v;
        }
    }

    /// self |= other & !minus
    pub fn or_minus(&mut self, other: &SimpleVob, minus: &SimpleVob) {
        assert_eq!(self.size, other.size);
        assert_eq!(self.size, minus.size);
        for ((slf, oth), mn) in self
            .data
            .iter_mut()
            .zip(other.data.iter())
            .zip(minus.data.iter())
        {
            *slf |= *oth & !*mn;
        }
    }

    pub fn and(&mut self, other: &SimpleVob) {
        assert_eq!(self.size, other.size);
        for (idx, v) in self.data.iter_mut().zip(other.data.iter()) {
            *idx &= *v;
        }
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|x| *x == 0)
    }

    pub fn and_is_zero(&self, other: &SimpleVob) -> bool {
        assert_eq!(self.size, other.size);
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| *a & *b == 0)
    }

    pub fn first_bit_set_here_and_in(&self, other: &SimpleVob) -> Option<usize> {
        assert_eq!(self.size, other.size);
        for (idx, (a, b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            let v = *a & *b;
            if v != 0 {
                return Some(idx * BITS + v.trailing_zeros() as usize);
            }
        }
        None
    }

    pub fn first_bit_set(&self) -> Option<usize> {
        for (idx, v) in self.data.iter().enumerate() {
            if *v != 0 {
                return Some(idx * BITS + v.trailing_zeros() as usize);
            }
        }
        None
    }
}

pub struct SimpleVobIter<'a> {
    vob: &'a SimpleVob,
    idx: usize,
}

impl<'a> Iterator for SimpleVobIter<'a> {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut bitoff = self.idx % BITS;
        let mut dataoff = self.idx / BITS;
        let data = &self.vob.data;
        while dataoff < data.len() {
            let d = data[dataoff] >> bitoff;
            if d != 0 {
                let idx = dataoff * BITS + d.trailing_zeros() as usize + bitoff;
                self.idx = idx + 1;
                return Some(idx as u32);
            }
            bitoff = 0;
            dataoff += 1;
        }
        None
    }
}

impl Index<usize> for SimpleVob {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        if self.is_allowed(index as TokenId) {
            &true
        } else {
            &false
        }
    }
}
