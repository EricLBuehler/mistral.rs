use regex_automata::{
    dfa::{dense, Automaton},
    util::syntax,
};
use rustc_hash::FxHashMap;
use std::{hash::Hash, vec};
use vob::{vob, Vob};

pub type PatIdx = usize;
pub type StateID = regex_automata::util::primitives::StateID;
use tracing::debug;

const LOG_LEXER: bool = false;

// enabling this is slightly faster, but it requires ~ |lexer_states|*|parser_states| bits
const PRECOMPUTE_AND: bool = false;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct LexerState {
    pub state: StateID,
    pub reachable: VobIdx,
}

impl LexerState {
    fn fake() -> Self {
        LexerState {
            state: StateID::default(),
            reachable: VobIdx::all_zero(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct VobIdx {
    v: u32,
}

impl VobIdx {
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(v: usize) -> Self {
        VobIdx { v: v as u32 }
    }

    pub fn all_zero() -> Self {
        VobIdx { v: 0 }
    }

    pub fn as_usize(&self) -> usize {
        self.v as usize
    }

    pub fn is_zero(&self) -> bool {
        self.v == 0
    }
}

#[derive(Clone)]
pub struct VobSet {
    vobs: Vec<Vob>,
    by_vob: FxHashMap<Vob, VobIdx>,
    non_empty: Vob,
}

impl VobSet {
    pub fn new() -> Self {
        VobSet {
            vobs: Vec::new(),
            by_vob: FxHashMap::default(),
            non_empty: Vob::new(),
        }
    }

    pub fn get(&mut self, vob: &Vob) -> VobIdx {
        if let Some(idx) = self.by_vob.get(vob) {
            return *idx;
        }
        let len = self.vobs.len();
        if len == 0 && !vob_is_zero(vob) {
            panic!("first vob must be empty");
        }
        let idx = VobIdx::new(len);
        self.vobs.push(vob.clone());
        self.by_vob.insert(vob.clone(), idx);
        idx
    }

    pub fn resolve(&self, idx: VobIdx) -> &Vob {
        &self.vobs[idx.as_usize()]
    }

    pub fn and_is_zero(&self, a: VobIdx, b: VobIdx) -> bool {
        if PRECOMPUTE_AND {
            !self.non_empty[a.as_usize() * self.vobs.len() + b.as_usize()]
        } else {
            vob_and_is_zero(&self.vobs[a.as_usize()], &self.vobs[b.as_usize()])
        }
    }

    pub fn pre_compute(&mut self) {
        if PRECOMPUTE_AND {
            let l = self.vobs.len();
            self.non_empty.resize(l * l, false);
            for x in 0..self.vobs.len() {
                for y in 0..=x {
                    if !vob_and_is_zero(&self.vobs[x], &self.vobs[y]) {
                        self.non_empty.set(x * l + y, true);
                        self.non_empty.set(y * l + x, true);
                    }
                }
            }
            debug!(
                "vob set: {} VOBs, {} nonempty",
                self.vobs.len(),
                self.non_empty.len()
            );
        }
    }
}

#[derive(Clone)]
pub struct Lexer {
    dfa: dense::DFA<Vec<u32>>,
    initial: LexerState,
    vobidx_by_state_off: Vec<VobIdx>,
}

impl Lexer {
    pub fn from(patterns: Vec<String>, vobset: &mut VobSet) -> Self {
        // TIME: 4ms
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(regex_automata::MatchKind::All),
            )
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build_many(&patterns)
            .unwrap();

        debug!(
            "dfa: {} bytes, {} patterns",
            dfa.memory_usage(),
            patterns.len(),
        );
        if false {
            for p in &patterns {
                debug!("  {}", p)
            }
        }

        let anch = regex_automata::Anchored::Yes;

        let mut incoming = FxHashMap::default();
        let initial = dfa.universal_start_state(anch).unwrap();
        let mut todo = vec![initial];
        incoming.insert(initial, Vec::new());

        // TIME: 1.5ms
        while let Some(s) = todo.pop() {
            for b in 0..=255 {
                let s2 = dfa.next_state(s, b);
                incoming.entry(s2).or_insert_with(|| {
                    todo.push(s2);
                    Vec::new()
                });
                incoming.get_mut(&s2).unwrap().push(s);
            }
        }

        let states = incoming.keys().copied().collect::<Vec<_>>();
        let mut reachable_patterns = FxHashMap::default();

        for s in &states {
            let mut v = vob![false; patterns.len()];
            let s2 = dfa.next_eoi_state(*s);
            if dfa.is_match_state(s2) {
                for idx in 0..dfa.match_len(s2) {
                    let idx = dfa.match_pattern(s2, idx).as_usize();
                    v.set(idx, true);
                    if LOG_LEXER {
                        debug!("  match: {:?} {}", *s, patterns[idx])
                    }
                }
            }
            reachable_patterns.insert(*s, v);
        }

        // TIME: 20ms
        loop {
            let mut num_set = 0;

            for s in &states {
                let ours = reachable_patterns.get(s).unwrap().clone();
                for o in &incoming[s] {
                    let theirs = reachable_patterns.get(o).unwrap();
                    let mut tmp = ours.clone();
                    tmp |= theirs;
                    if tmp != *theirs {
                        num_set += 1;
                        reachable_patterns.insert(*o, tmp);
                    }
                }
            }

            if LOG_LEXER {
                debug!("iter {} {}", num_set, states.len());
            }
            if num_set == 0 {
                break;
            }
        }

        let mut states_idx = states.iter().map(|x| x.as_usize()).collect::<Vec<_>>();
        states_idx.sort();

        let shift = dfa.stride2();
        let mut vobidx_by_state_off =
            vec![VobIdx::all_zero(); 1 + (states_idx.iter().max().unwrap() >> shift)];
        for (k, v) in reachable_patterns.iter() {
            vobidx_by_state_off[k.as_usize() >> shift] = vobset.get(v);
        }

        debug!("initial: {:?}; {} states", initial, states.len());

        let mut lex = Lexer {
            dfa,
            vobidx_by_state_off,
            initial: LexerState::fake(),
        };

        lex.initial = lex.mk_state(initial);

        if LOG_LEXER {
            for s in &states {
                if lex.is_dead(*s) {
                    debug!("dead: {:?} {}", s, lex.dfa.is_dead_state(*s));
                }
            }

            debug!("reachable: {:#?}", reachable_patterns);
        }

        lex
    }

    pub fn file_start_state(&self) -> StateID {
        self.initial.state
        // pretend we've just seen a newline at the beginning of the file
        // TODO: this should be configurable
        // self.dfa.next_state(self.initial.state, b'\n')
    }

    fn mk_state(&self, state: StateID) -> LexerState {
        LexerState {
            state,
            reachable: self.reachable_tokens(state),
        }
    }

    fn is_dead(&self, state: StateID) -> bool {
        self.reachable_tokens(state).is_zero()
    }

    fn reachable_tokens(&self, state: StateID) -> VobIdx {
        self.vobidx_by_state_off[state.as_usize() >> self.dfa.stride2()]
    }

    fn get_token(&self, prev: StateID) -> Option<PatIdx> {
        let state = self.dfa.next_eoi_state(prev);
        if !self.dfa.is_match_state(state) {
            return None;
        }

        // we take the first token that matched
        // (eg., "while" will match both keyword and identifier, but keyword is first)
        let pat_idx = (0..self.dfa.match_len(state))
            .map(|idx| self.dfa.match_pattern(state, idx).as_usize())
            .min()
            .unwrap();

        if LOG_LEXER {
            debug!("token: {}", pat_idx);
        }

        Some(pat_idx)
    }

    #[inline(always)]
    pub fn advance(&self, prev: StateID, byte: Option<u8>) -> Option<(LexerState, Option<PatIdx>)> {
        let dfa = &self.dfa;
        if let Some(byte) = byte {
            let state = dfa.next_state(prev, byte);
            if LOG_LEXER {
                debug!(
                    "lex: {:?} -{:?}-> {:?} d={}",
                    prev,
                    byte as char,
                    state,
                    self.is_dead(state),
                );
            }
            let v = self.reachable_tokens(state);
            if v.is_zero() {
                // if final_state is a match state, find the token that matched
                let tok = self.get_token(prev);
                if tok.is_none() {
                    None
                } else {
                    let state = dfa.next_state(self.initial.state, byte);
                    if LOG_LEXER {
                        debug!("lex0: {:?} -{:?}-> {:?}", self.initial, byte as char, state);
                    }
                    Some((self.mk_state(state), tok))
                }
            } else {
                Some((
                    LexerState {
                        state,
                        reachable: v,
                    },
                    None,
                ))
            }
        } else {
            let tok = self.get_token(prev);
            if tok.is_none() {
                None
            } else {
                Some((self.initial, tok))
            }
        }
    }
}

fn vob_and_is_zero(a: &Vob, b: &Vob) -> bool {
    debug_assert!(a.len() == b.len());
    for (a, b) in a.iter_storage().zip(b.iter_storage()) {
        if a & b != 0 {
            return false;
        }
    }
    true
}

fn vob_is_zero(v: &Vob) -> bool {
    for b in v.iter_storage() {
        if b != 0 {
            return false;
        }
    }
    true
}
