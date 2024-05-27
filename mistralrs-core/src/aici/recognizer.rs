use crate::aici::toktree::{Recognizer, SpecialToken};
use std::fmt::Debug;

pub trait FunctionalRecognizer<S: Copy> {
    /// Initial state
    fn initial(&self) -> Result<S, anyhow::Error>;
    /// Extend the recognizer with given byte.
    fn append(&self, state: S, byte: u8) -> S;
    /// Check if given byte is allowed in given state.
    fn byte_allowed(&self, state: S, byte: u8) -> bool;
    /// Check if given special token is allowed in given state.
    fn special_allowed(&self, state: S, tok: SpecialToken) -> bool;
}

#[derive(Clone)]
pub struct StackRecognizer<S: Copy, R: FunctionalRecognizer<S>> {
    rec: R,
    stack: Vec<S>,
    stack_ptr: usize,
}

impl<S: Copy, R: FunctionalRecognizer<S>> StackRecognizer<S, R> {
    pub fn from(rec: R) -> anyhow::Result<Self> {
        let stack = vec![rec.initial()?; 130];
        let rec = StackRecognizer {
            rec,
            stack,
            stack_ptr: 0,
        };
        Ok(rec)
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.stack_ptr = 0;
        self.stack[0] = self.rec.initial()?;
        Ok(())
    }
}

impl<S: Copy + Debug, R: FunctionalRecognizer<S>> Debug for StackRecognizer<S, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackRecognizer")
            .field("top", &self.stack[self.stack_ptr])
            .finish()
    }
}

impl<S: Copy + Debug, R: FunctionalRecognizer<S>> Recognizer for StackRecognizer<S, R> {
    #[inline(always)]
    fn push_byte(&mut self, byte: u8) {
        let state = self.stack[self.stack_ptr];
        let state = self.rec.append(state, byte);
        self.stack_ptr += 1;
        self.stack[self.stack_ptr] = state;
    }

    #[inline(always)]
    fn pop_bytes(&mut self, num: usize) {
        self.stack_ptr -= num;
    }

    #[inline(always)]
    fn byte_allowed(&mut self, byte: u8) -> bool {
        self.rec.byte_allowed(self.stack[self.stack_ptr], byte)
    }

    fn trie_finished(&mut self) {
        assert!(self.stack_ptr == 0);
    }

    fn collapse(&mut self) {
        self.stack[0] = self.stack[self.stack_ptr];
        self.stack_ptr = 0;
    }

    fn special_allowed(&mut self, tok: SpecialToken) -> bool {
        self.rec.special_allowed(self.stack[self.stack_ptr], tok)
    }

    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if self.rec.byte_allowed(self.stack[self.stack_ptr], byte) {
            self.push_byte(byte);
            true
        } else {
            false
        }
    }
}
