// use 8:24 encoding - num_ch:tok_id (ch_byte:ch_off)* - 8 bytes per tree node
// special case num_ch=0xff -> num_ch=0x100

use rustc_hash::FxHashMap;

use crate::aici::{
    bytes::{
        box_from_bytes, clone_as_bytes, clone_vec_as_bytes, to_hex_string, vec_from_bytes,
        TokRxInfo, TokenId,
    },
    svob::SimpleVob,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    Unknown,
    Padding,
    Separator,
    BeginningOfSentence,
    EndOfSentence,
}

pub trait Recognizer {
    /// If `stack.top()` transitions via `byte` to `X`, execute `stack.push(X)`.
    fn push_byte(&mut self, byte: u8) {
        if !self.try_push_byte(byte) {
            panic!("byte {:?} not allowed", byte as char)
        }
    }
    /// for _ in 0..num { stack.pop() }
    fn pop_bytes(&mut self, num: usize);
    /// X = stack.top(); stack.empty(); stack.push(X)
    fn collapse(&mut self);
    /// check if stack.top() transitions via byte to a viable state
    fn byte_allowed(&mut self, byte: u8) -> bool {
        if self.try_push_byte(byte) {
            self.pop_bytes(1);
            true
        } else {
            false
        }
    }
    /// check if stack.top() transitions via tok to a viable state
    fn special_allowed(&mut self, tok: SpecialToken) -> bool;
    /// Called when iteration over the trie is finished
    /// Stack has exactly one element then.
    fn trie_finished(&mut self);
    /// Called when iteration over the trie is started
    fn trie_started(&mut self) {}
    /// This combines `push_byte` and `byte_allowed` into one function for performance.
    fn try_push_byte(&mut self, byte: u8) -> bool;
}

pub struct TokTrie {
    info: TokRxInfo,
    token_offsets: Vec<u32>,
    token_data: Vec<u8>,
    nodes: Vec<TrieNode>,
    max_token_len: usize,
    token_duplicates: FxHashMap<TokenId, Vec<TokenId>>,
}

#[repr(C)]
pub struct TokTrieHeader {
    magic: u32,
    hd_size: u32,
    trie_bytes: u32,
    token_offset_bytes: u32,
    token_data_bytes: u32,
    info: TokRxInfo,
    align: [u32; 0],
}

impl TokTrieHeader {
    const MAGIC: u32 = 0x558b6fd3;
}

#[repr(C)]
pub struct TrieNode {
    // byte:token
    bits: u32,
    bits2: u32,
}

const NO_TOKEN: u32 = 0xffffff;

impl TrieNode {
    fn new(byte: u8, token_id: u32, num_parents: u8) -> TrieNode {
        TrieNode {
            bits: (token_id << 8) | byte as u32,
            bits2: num_parents as u32,
        }
    }

    #[inline(always)]
    pub fn byte(&self) -> u8 {
        (self.bits & 0xff) as u8
    }

    #[inline(always)]
    pub fn subtree_size(&self) -> usize {
        (self.bits2 >> 8) as usize
    }

    #[inline(always)]
    pub fn num_parents(&self) -> usize {
        (self.bits2 & 0xff) as usize
    }

    #[inline(always)]
    pub fn token_id(&self) -> Option<u32> {
        let r = self.bits >> 8;
        if r == NO_TOKEN {
            None
        } else {
            Some(r)
        }
    }
}

impl TokTrie {
    pub fn from(info: &TokRxInfo, words: &Vec<Vec<u8>>) -> Self {
        let mut trie = TrieHash::new(0xff);
        let mut token_offsets = Vec::new();
        let mut token_data = Vec::new();
        assert!(info.vocab_size == words.len() as u32);
        for (idx, word) in words.iter().enumerate() {
            if word.len() > 0 {
                trie.insert(word, idx as u32);
            }
            assert!(word.len() < 0xff);
            let desc = (word.len() as u32) | ((token_data.len() as u32) << 8);
            token_offsets.push(desc);
            token_data.extend_from_slice(word);
        }
        let mut nodes = Vec::new();
        trie.serialize(&mut nodes, 0);
        let mut r = TokTrie {
            info: info.clone(),
            token_offsets,
            token_data,
            nodes,
            max_token_len: 0,
            token_duplicates: FxHashMap::default(),
        };
        r.finalize_ctor();
        r
    }

    fn finalize_ctor(&mut self) {
        for tok_id in 0..self.info.vocab_size {
            let bytes = self.token(tok_id);
            let tok_ids = self.greedy_tokenize(bytes);
            self.max_token_len = std::cmp::max(self.max_token_len, bytes.len());
            if tok_ids.len() == 1 && tok_ids[0] != tok_id {
                self.token_duplicates
                    .entry(tok_ids[0])
                    .or_insert_with(Vec::new)
                    .push(tok_id);
            }
        }
        self.validate();
    }

    fn node_offset(&self, n: &TrieNode) -> usize {
        let off = unsafe { (n as *const TrieNode).offset_from(self.root() as *const TrieNode) };
        assert!(off >= 0);
        let off = off as usize;
        assert!(off < self.nodes.len());
        off
    }

    fn next_node(&self, n: &TrieNode) -> usize {
        return self.node_offset(n) + n.subtree_size();
    }

    pub fn info(&self) -> &TokRxInfo {
        &self.info
    }

    pub fn special_token(&self, tok: SpecialToken) -> TokenId {
        match tok {
            SpecialToken::EndOfSentence => self.info.tok_eos,
            _ => panic!("non-EOS special_token() called"), // TODO?
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.info.vocab_size as usize
    }

    pub fn alloc_token_set(&self) -> SimpleVob {
        let mut r = SimpleVob::new();
        r.resize(self.vocab_size() + 1);
        r
    }

    pub fn token_set_dbg(&self, ts: &SimpleVob) -> String {
        let max_examples = 50;

        let ts_neg = ts.negated(self.vocab_size());
        let use_neg = ts_neg.num_set() * 20 < ts.num_set();
        let ts1 = if use_neg { &ts_neg } else { &ts };
        let num_set = ts1.num_set();
        let max_tok = std::cmp::min(max_examples, num_set);
        let mut token_names = Vec::new();
        for idx in 0..self.vocab_size() {
            if ts1.is_allowed(idx as TokenId) {
                token_names.push(self.token_dbg(idx as TokenId));
                if token_names.len() >= max_tok {
                    break;
                }
            }
        }
        if token_names.len() < num_set {
            token_names.push("...".to_string());
        }
        format!(
            "TokenSet: {}/{}; {}{}",
            ts.num_set(),
            self.vocab_size(),
            if use_neg { "ALL EXCEPT " } else { "" },
            token_names.join(", ")
        )
    }

    pub fn alloc_logits(&self) -> Vec<f32> {
        vec![0.0; self.vocab_size() + 1]
    }

    pub fn tokens_dbg(&self, toks: &[u32]) -> String {
        let minimal = false;
        let sep = "‧";
        let joined = toks
            .iter()
            .map(|t| {
                let s = self.token_dbg(*t);
                if s.starts_with("\"") {
                    let inner = s[1..s.len() - 1].to_string();
                    let b = s.as_bytes();
                    // for " [\w]..." and " " the sep in front is implicit
                    if minimal && b[1] == b' ' && ((b[2] as char).is_alphanumeric() || b.len() == 3)
                    {
                        inner
                    } else {
                        format!("{}{}", sep, inner)
                    }
                } else {
                    format!("≺{}≻", s)
                }
            })
            .collect::<Vec<_>>()
            .join("");
        format!("\"{}\"", joined.trim_start_matches(sep))
    }

    pub fn token_dbg(&self, idx: u32) -> String {
        if idx == self.info.tok_eos {
            "EOS".to_string()
        } else if idx as usize >= self.vocab_size() {
            format!("OOB[{}]", idx)
        } else {
            // format!("{:?}[{}]", self.token_str(idx), idx)
            let s = self.token_str(idx);
            if s.len() == 0 {
                format!("EMPTY[{}]", idx)
            } else if !s.contains('\u{fffd}') {
                format!("{:?}", s)
            } else {
                let bytes = self.token(idx);
                format!("HEX[{}]", to_hex_string(bytes))
            }
        }
    }

    pub fn token_str(&self, idx: u32) -> String {
        String::from_utf8_lossy(self.token(idx)).to_string()
    }

    pub fn token(&self, idx: u32) -> &[u8] {
        let off = self.token_offsets[idx as usize];
        let len = off & 0xff;
        let off = (off >> 8) as usize;
        &self.token_data[off..(off + len as usize)]
    }

    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        tokens
            .iter()
            .flat_map(|t| self.token(*t).to_vec())
            .collect()
    }

    pub fn decode_str(&self, tokens: &[TokenId]) -> String {
        String::from_utf8_lossy(&self.decode(tokens)).to_string()
    }

    pub fn greedy_tokenize(&self, bytes: &[u8]) -> Vec<TokenId> {
        let mut r = Vec::new();
        if bytes.len() == 0 {
            return r;
        }

        let mut n = self.root();
        let mut last_tok = None;
        let mut last_idx = 0;
        let mut idx = 0;
        while idx < bytes.len() {
            match self.child_at_byte(n, bytes[idx]) {
                Some(c) => {
                    if let Some(tok) = c.token_id() {
                        last_tok = Some(tok);
                        last_idx = idx;
                    }
                    n = c;
                }
                None => {
                    r.push(last_tok.unwrap());
                    idx = last_idx;
                    n = self.root();
                }
            }
            idx = idx + 1;
        }
        r.push(last_tok.unwrap());
        r
    }

    pub fn has_extensions(&self, bytes: &[u8]) -> bool {
        match self.child_at_bytes(self.root(), bytes) {
            None => false,
            Some(n) => n.subtree_size() > 1,
        }
    }

    pub fn token_id(&self, bytes: &[u8]) -> Option<TokenId> {
        let (tok, len) = self.prefix_token_id(bytes);
        // println!("tok_id {:?} {:?} {:?} ", bytes, tok, len);
        if len == bytes.len() {
            Some(tok)
        } else {
            None
        }
    }

    pub fn prefix_token_id(&self, bytes: &[u8]) -> (TokenId, usize) {
        assert!(bytes.len() > 0);
        let mut last = (0, 0);
        let mut n = self.root();
        for (idx, byte) in bytes.iter().enumerate() {
            n = match self.child_at_byte(n, *byte) {
                Some(n) => n,
                None => break,
            };
            if let Some(tok) = n.token_id() {
                last = (tok, idx + 1);
            }
        }
        return last;
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let pref = std::mem::size_of::<TokTrieHeader>();
        let hd = *box_from_bytes::<TokTrieHeader>(&bytes[0..pref]);
        assert!(hd.magic == TokTrieHeader::MAGIC);
        assert!(hd.hd_size as usize == pref);

        let trie_end = pref + hd.trie_bytes as usize;
        let nodes = vec_from_bytes(&bytes[pref..trie_end]);
        let offsets_end = trie_end + hd.token_offset_bytes as usize;
        let token_offsets = vec_from_bytes(&bytes[trie_end..offsets_end]);
        let token_data = vec_from_bytes(&bytes[offsets_end..]);

        let mut r = TokTrie {
            info: hd.info,
            token_offsets,
            token_data,
            nodes,
            max_token_len: 0,
            token_duplicates: FxHashMap::default(),
        };
        r.finalize_ctor();
        r
    }

    pub fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    fn validate_node(&self, n: &TrieNode, ep: usize, used: &mut [bool]) {
        if let Some(tok) = n.token_id() {
            assert!(tok < self.info.vocab_size);
            assert!(!used[tok as usize]);
            used[tok as usize] = true;
        }
        let endp = self.next_node(n);
        assert!(endp <= ep);
        for child in self.node_children(n) {
            self.validate_node(child, endp, used);
        }
    }

    fn validate(&self) {
        self.validate_node(
            self.root(),
            self.next_node(self.root()),
            &mut vec![false; self.info.vocab_size as usize],
        );
        for idx in 0..self.info.vocab_size {
            let _ = self.token(idx);
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut trie_data = clone_vec_as_bytes(&self.nodes);
        let mut token_offsets = clone_vec_as_bytes(&self.token_offsets);
        let mut token_data = clone_vec_as_bytes(&self.token_data);

        let hd = TokTrieHeader {
            magic: TokTrieHeader::MAGIC,
            hd_size: std::mem::size_of::<TokTrieHeader>() as u32,
            trie_bytes: trie_data.len() as u32,
            token_offset_bytes: token_offsets.len() as u32,
            token_data_bytes: trie_data.len() as u32,
            info: self.info.clone(),
            align: [],
        };

        let mut bytes = clone_as_bytes(&hd);
        bytes.append(&mut trie_data);
        bytes.append(&mut token_offsets);
        bytes.append(&mut token_data);
        bytes
    }

    pub fn root(&self) -> &TrieNode {
        &self.nodes[0]
    }

    pub fn check_against(&self, tokens: &Vec<Vec<u8>>) {
        let vocab_size = tokens.len();
        for idx in 0..vocab_size {
            let bytes = &tokens[idx];
            let tid = idx as TokenId;
            assert!(bytes == self.token(tid));
            let root = self.root();
            if bytes.len() > 0 {
                let tid2 = self
                    .child_at_bytes(root, &bytes)
                    .unwrap()
                    .token_id()
                    .unwrap();
                if tid != tid2 {
                    assert!(self.token_duplicates[&tid2].contains(&tid));
                }
            }
        }
    }

    pub fn child_at_byte<'a>(&'a self, n: &'a TrieNode, byte: u8) -> Option<&'a TrieNode> {
        for child in self.node_children(n) {
            if child.byte() == byte {
                return Some(child);
            }
        }
        None
    }

    pub fn node_children(&self, n: &TrieNode) -> NodeChildren {
        let off = self.node_offset(n);
        NodeChildren {
            trie: self,
            current_offset: off + 1,
            end_offset: off + n.subtree_size(),
        }
    }

    pub fn child_at_bytes<'a>(&'a self, mut n: &'a TrieNode, bytes: &[u8]) -> Option<&'a TrieNode> {
        for &byte in bytes {
            n = match self.child_at_byte(n, byte) {
                Some(n) => n,
                None => return None,
            }
        }
        Some(n)
    }

    pub fn compute_bias(&self, r: &mut impl Recognizer, logits: &mut SimpleVob) {
        self.compute_bias_ext(r, logits, &[]);
    }

    pub fn compute_bias_ext(&self, r: &mut impl Recognizer, logits: &mut SimpleVob, start: &[u8]) {
        logits.set_all(false);
        for tok in vec![SpecialToken::EndOfSentence] {
            if r.special_allowed(tok) {
                logits.allow_token(self.special_token(tok))
            }
        }
        // all prefixes of 'start' are also allowed
        if start.len() > 0 {
            for len in 1..start.len() - 1 {
                let bytes = &start[0..len];
                if let Some(tok) = self.token_id(bytes) {
                    logits.allow_token(tok);
                }
            }
        }
        self.add_bias(r, logits, start);
        self.apply_duplicates(logits);
    }

    pub fn apply_duplicates(&self, logits: &mut SimpleVob) {
        for (tok, dups) in &self.token_duplicates {
            if logits.is_allowed(*tok) {
                for &dup in dups {
                    logits.allow_token(dup);
                }
            }
        }
    }

    pub fn append_tokens(&self, r: &mut impl Recognizer, ts: &[TokenId]) {
        for t in ts {
            self.append_token(r, *t)
        }
    }

    pub fn append_token(&self, r: &mut impl Recognizer, t: TokenId) {
        // println!("append_token: {}", self.token_dbg(t));
        let bytes = self.token(t);
        for &byte in bytes {
            r.push_byte(byte)
        }
        r.collapse()
    }

    pub fn token_allowed(&self, r: &mut impl Recognizer, t: TokenId) -> bool {
        let bytes = self.token(t);
        let mut num = 0;
        let mut ok = true;
        for &byte in bytes {
            if r.try_push_byte(byte) {
                num += 1;
            } else {
                ok = false;
                break;
            }
        }
        r.pop_bytes(num);
        ok
    }

    #[inline(never)]
    pub fn add_bias(&self, r: &mut impl Recognizer, toks: &mut SimpleVob, start: &[u8]) {
        r.trie_started();
        let n = self.child_at_bytes(self.root(), start).unwrap();
        let defl_tok = self.vocab_size() as u32;
        let off = self.node_offset(n);
        let mut p = off + 1;
        let endp = off + n.subtree_size();
        while p < endp {
            let n = &self.nodes[p];
            let b = n.byte();
            if r.try_push_byte(b) {
                toks.allow_token(n.token_id().unwrap_or(defl_tok));
                r.pop_bytes(if n.subtree_size() == 1 {
                    n.num_parents()
                } else {
                    0
                });

                p += 1;
            } else {
                p += n.subtree_size();
                r.pop_bytes(n.num_parents() - 1);
            }
        }
        r.trie_finished();
        // revert the fake token
        toks.disallow_token(defl_tok);
    }
}

pub struct NodeChildren<'a> {
    trie: &'a TokTrie,
    current_offset: usize,
    end_offset: usize,
}

impl<'a> Iterator for NodeChildren<'a> {
    type Item = &'a TrieNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset < self.end_offset {
            let node = &self.trie.nodes[self.current_offset];
            self.current_offset += node.subtree_size();
            Some(node)
        } else {
            None
        }
    }
}

struct TrieHash {
    token_id: u32,
    byte: u8,
    children: Vec<TrieHash>,
}

impl TrieHash {
    fn new(byte: u8) -> TrieHash {
        TrieHash {
            token_id: NO_TOKEN,
            byte,
            children: Vec::new(),
        }
    }
    fn insert(&mut self, word: &[u8], token_id: u32) {
        if word.len() == 0 {
            // Some tokenizers have duplicate tokens...
            // we just override
            // assert!(self.token_id == NO_TOKEN);
            self.token_id = token_id;
        } else {
            if self.children.len() == 0x100 {
                // assert!(self.children[word[0] as usize].byte == word[0]);
                self.children[word[0] as usize].insert(&word[1..], token_id);
                return;
            }

            for ch in &mut self.children {
                if ch.byte == word[0] {
                    ch.insert(&word[1..], token_id);
                    return;
                }
            }

            let mut ch = TrieHash::new(word[0]);
            ch.insert(&word[1..], token_id);
            self.children.push(ch);

            // if it's getting dense, make it full
            // for cl100k threshold 60->15 nodes, 50->22, 40->45 30->94
            // for llama (32k) 50->5, 40->15
            // TODO remove this?
            if self.children.len() > 250 {
                let mut v2 = (0..=255).map(TrieHash::new).collect::<Vec<_>>();
                for ch in self.children.drain(..) {
                    let idx = ch.byte as usize;
                    v2[idx] = ch;
                }
                self.children = v2;
            }
        }
    }
    fn serialize(&mut self, data: &mut Vec<TrieNode>, num_parents: u8) {
        let idx = data.len();
        let mut num_ch = self.children.len();
        data.push(TrieNode::new(self.byte, self.token_id, num_parents));
        self.children.sort_by_key(|e| e.byte);
        for entry in &mut self.children {
            num_ch -= 1;
            entry.serialize(data, if num_ch == 0 { num_parents + 1 } else { 1 });
        }
        data[idx].bits2 |= ((data.len() - idx) as u32) << 8;
    }
}
