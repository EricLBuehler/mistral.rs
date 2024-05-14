use crate::aici::lex::{Lexer, LexerState, StateID, VobIdx, VobSet};
use crate::aici::toktree::{Recognizer, SpecialToken};
use anyhow::Result;
use cfgrammar::{
    yacc::{YaccGrammar, YaccKind},
    Span, Spanned, Symbol, TIdx,
};
use lrtable::{from_yacc, Action, Minimiser, StIdx, StateTable};
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};
use std::vec;
use tracing::debug;
use vob::{vob, Vob};

type StorageT = u32;
type PStack<StorageT> = Vec<StIdx<StorageT>>; // Parse stack

const LOG_PARSER: bool = false;

#[derive(Debug, Clone, Copy)]
enum ParseResult {
    Accept,
    Error,
    Continue,
}

#[derive(Clone)]
struct CfgStats {
    yacc_actions: usize,
    states_pushed: usize,
}

pub struct CfgParser {
    grm: Arc<YaccGrammar<StorageT>>,
    stable: Arc<StateTable<StorageT>>,
    lexer: Lexer,
    byte_states: Vec<ByteState>,
    pat_idx_to_tidx: Vec<TIdx<u32>>,
    vobset: VobSet,
    stats: RwLock<CfgStats>,
    tidx_to_pat_idx: FxHashMap<TIdx<u32>, usize>,
    parse_stacks: Vec<Vec<StIdx<u32>>>,
    skip_patterns: Vob,
    friendly_pattern_names: Vec<String>,
    viable_vobidx_by_state: Vec<VobIdx>,
}

fn is_rx(name: &str) -> bool {
    name.len() > 2 && name.starts_with('/') && name.ends_with('/')
}

fn quote_rx(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_digit()
                || ch.is_ascii_lowercase()
                || ch.is_ascii_uppercase()
                || '<' == ch
                || '>' == ch
            {
                ch.to_string()
            } else {
                format!("\\{}", ch)
            }
        })
        .collect::<String>()
}

pub(crate) fn parse_rx_token(name: &str) -> String {
    if is_rx(name) {
        name[1..name.len() - 1].to_string()
    } else {
        quote_rx(name)
    }
}

fn span_to_str(s: &Span, src: &str) -> String {
    let mut line = 1;
    let mut last_nl = 0;
    for (idx, ch) in src.chars().enumerate() {
        if idx == s.start() {
            break;
        }
        if ch == '\n' {
            line += 1;
            last_nl = idx;
        }
    }
    let column = s.start() - last_nl;
    format!("({},{})", line, column)
}

pub(crate) fn parse_yacc(yacc: &str) -> Result<YaccGrammar> {
    let grmkind = YaccKind::Original(cfgrammar::yacc::YaccOriginalActionKind::NoAction);
    let grm = match YaccGrammar::new(grmkind, yacc) {
        Ok(grm) => grm,
        Err(e) => {
            let err_str = e
                .iter()
                .map(|e| {
                    let spans = e
                        .spans()
                        .iter()
                        .map(|s| span_to_str(s, yacc))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}: {}", spans, e)
                })
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!("yacc grammar errors:\n{}", err_str);
        }
    };
    Ok(grm)
}

impl CfgParser {
    pub fn from_yacc(yacc: &str) -> Result<Self> {
        let grm = parse_yacc(yacc)?;
        // TIME: all these annotation are for native release x86 build for C grammar
        // TIME: 27ms
        let (sgraph, stable) = match from_yacc(&grm, Minimiser::Pager) {
            Ok(r) => r,
            Err(e) => {
                if false {
                    // not sure this works:
                    anyhow::bail!("state table error:\n{e} on {:?}", grm.action(e.pidx));
                }
                anyhow::bail!("state table error:\n{e}");
            }
        };

        if false {
            debug!("core\n{}\n\n", sgraph.pp(&grm, true));
            for pidx in grm.iter_pidxs() {
                let prod = grm.prod(pidx);
                debug!("{:?} -> {}", prod, prod.len());
            }
        }

        let mut pat_idx_to_tidx = grm
            .iter_tidxs()
            .filter(|tidx| grm.token_name(*tidx).is_some())
            .collect::<Vec<_>>();

        pat_idx_to_tidx.sort_by_key(|tidx| {
            let name = grm.token_name(*tidx).unwrap();
            let l = name.len() as isize;
            if is_rx(name) {
                -l + 100000
            } else {
                -l
            }
        });

        let patterns = pat_idx_to_tidx
            .iter()
            .map(|tok| parse_rx_token(grm.token_name(*tok).unwrap()))
            .collect::<Vec<_>>();

        let mut tidx_to_pat_idx = FxHashMap::default();
        for (idx, _tok) in patterns.iter().enumerate() {
            tidx_to_pat_idx.insert(pat_idx_to_tidx[idx], idx);
        }

        let mut skip_patterns = vob![false; patterns.len()];
        let friendly_pattern_names = pat_idx_to_tidx
            .iter()
            .map(|tok| grm.token_name(*tok).unwrap().to_string())
            .collect::<Vec<_>>();

        for ridx in grm.iter_rules() {
            let rname = grm.rule_name_str(ridx);
            if rname.to_uppercase() != rname {
                continue;
            }
            for pidx in grm.rule_to_prods(ridx) {
                let toks = grm.prod(*pidx);
                if let [Symbol::Token(tidx)] = toks {
                    let idx = *tidx_to_pat_idx.get(tidx).unwrap();
                    // this doesn't seem very useful
                    // friendly_pattern_names[idx] = rname.to_string();
                    if rname == "SKIP" {
                        skip_patterns.set(idx, true);
                    }
                }
            }
        }

        debug!("patterns: {:?}", friendly_pattern_names);

        let mut vobset = VobSet::new();
        // all-zero has to be inserted first
        let _all0 = vobset.get(&vob![false; patterns.len()]);
        let all1 = vobset.get(&vob![true; patterns.len()]);

        // TIME: 27ms
        let dfa = Lexer::from(patterns, &mut vobset);

        let cfg_start = stable.start_state();
        let parse_stacks = vec![vec![cfg_start]];

        let byte_state = ByteState {
            lexer_state: dfa.file_start_state(),
            parse_stack_idx: PStackIdx(0),
            viable: all1,
        };

        let viable_vobidx_by_state = sgraph
            .iter_stidxs()
            .enumerate()
            .map(|(idx, stidx)| {
                assert!(idx == stidx.as_storaget() as usize);

                // skip patterns (whitespace) are always viable
                let mut r = skip_patterns.clone();
                for tidx in stable.state_actions(stidx) {
                    match stable.action(stidx, tidx) {
                        Action::Error => {}
                        _ => {
                            if let Some(pat_idx) = tidx_to_pat_idx.get(&tidx) {
                                r.set(*pat_idx, true);
                            }
                        }
                    }
                }

                vobset.get(&r)
            })
            .collect::<Vec<_>>();

        let mut cfg = CfgParser {
            grm: grm.into(),
            stable: stable.into(),
            lexer: dfa,
            byte_states: vec![byte_state],
            pat_idx_to_tidx,
            tidx_to_pat_idx,
            viable_vobidx_by_state,
            skip_patterns,
            friendly_pattern_names,
            parse_stacks,
            vobset,
            stats: RwLock::new(CfgStats {
                yacc_actions: 0,
                states_pushed: 0,
            }),
        };

        cfg.vobset.pre_compute();

        // compute viable set of initial tokens
        cfg.byte_states[0].viable = cfg.viable_vobidx(cfg_start);
        if LOG_PARSER {
            debug!(
                "initial viable: {:?}",
                cfg.vobset.resolve(cfg.byte_states[0].viable)
            );
        }

        Ok(cfg)
    }

    fn viable_vobidx(&self, stidx: StIdx<StorageT>) -> VobIdx {
        self.viable_vobidx_by_state[stidx.as_storaget() as usize]
    }

    #[allow(dead_code)]
    fn friendly_token_name(&self, lexeme: TIdx<StorageT>) -> &str {
        if let Some(pidx) = self.tidx_to_pat_idx.get(&lexeme) {
            &self.friendly_pattern_names[*pidx]
        } else if self.grm.eof_token_idx() == lexeme {
            return "<EOF>";
        } else {
            return "<???>";
        }
    }

    fn parse_lexeme(&self, lexeme: TIdx<StorageT>, pstack: &mut PStack<StorageT>) -> ParseResult {
        loop {
            let stidx = *pstack.last().unwrap();

            let act = self.stable.action(stidx, lexeme);

            if LOG_PARSER {
                debug!(
                    "parse: {:?} {:?} -> {:?}",
                    pstack,
                    self.friendly_token_name(lexeme),
                    act
                );
            }

            match act {
                Action::Reduce(pidx) => {
                    let ridx = self.grm.prod_to_rule(pidx);
                    let pop_idx = pstack.len() - self.grm.prod(pidx).len();
                    pstack.drain(pop_idx..);
                    let prior = *pstack.last().unwrap();
                    pstack.push(self.stable.goto(prior, ridx).unwrap());
                }
                Action::Shift(state_id) => {
                    pstack.push(state_id);
                    return ParseResult::Continue;
                }
                Action::Accept => {
                    // only happens when lexeme is EOF
                    return ParseResult::Accept;
                }
                Action::Error => {
                    return ParseResult::Error;
                }
            }
        }
    }

    #[allow(dead_code)]
    fn print_viable(&self, lbl: &str, vob: &Vob) {
        debug!("viable tokens {}:", lbl);
        for (idx, b) in vob.iter().enumerate() {
            if b {
                debug!("  {}: {}", idx, self.friendly_pattern_names[idx]);
            }
        }
    }

    // None means EOF
    #[inline(always)]
    fn try_push(&mut self, byte: Option<u8>) -> Option<ByteState> {
        let top = self.byte_states.last().unwrap().clone();
        if LOG_PARSER {
            print!("try_push[{}]: ", self.byte_states.len());
            if let Some(b) = byte {
                print!("{:?}", b as char)
            } else {
                print!("<EOF>")
            }
        }
        let (info, res) = match self.lexer.advance(top.lexer_state, byte) {
            // Error?
            None => ("lex-err", None),
            // Just new state, no token - the hot path
            Some((ls, None)) => (
                "lex",
                self.mk_byte_state(ls, top.parse_stack_idx, top.viable),
            ),
            // New state and token generated
            Some((ls, Some(pat_idx))) => ("parse", self.run_parser(pat_idx, &top, ls)),
        };
        if LOG_PARSER {
            debug!(
                " -> {} {}",
                info,
                if res.is_none() { "error" } else { "ok" }
            );
        }
        res
    }

    fn pstack_for(&self, top: &ByteState) -> &PStack<StorageT> {
        &self.parse_stacks[top.parse_stack_idx.0]
    }

    fn push_pstack(&mut self, top: &ByteState, pstack: Vec<StIdx<u32>>) -> PStackIdx {
        let new_idx = PStackIdx(top.parse_stack_idx.0 + 1);
        if self.parse_stacks.len() <= new_idx.0 {
            self.parse_stacks.push(Vec::new());
        }
        self.parse_stacks[new_idx.0] = pstack;
        new_idx
    }

    fn run_parser(&mut self, pat_idx: usize, top: &ByteState, ls: LexerState) -> Option<ByteState> {
        {
            let mut s = self.stats.write().unwrap();
            s.yacc_actions += 1;
        }

        let pstack = self.pstack_for(top);
        if self.skip_patterns[pat_idx] {
            let stidx = *pstack.last().unwrap();
            let viable = self.viable_vobidx(stidx);
            //self.print_viable("reset", &viable);
            if LOG_PARSER {
                debug!("parse: {:?} skip", pstack);
            }
            // reset viable states - they have been narrowed down to SKIP
            self.mk_byte_state(ls, top.parse_stack_idx, viable)
        } else {
            let tidx = self.pat_idx_to_tidx[pat_idx];
            let mut pstack = pstack.clone();
            match self.parse_lexeme(tidx, &mut pstack) {
                ParseResult::Accept => panic!("accept non EOF?"),
                ParseResult::Continue => {
                    let stidx = *pstack.last().unwrap();
                    let viable = self.viable_vobidx(stidx);
                    let new_idx = self.push_pstack(top, pstack);
                    self.mk_byte_state(ls, new_idx, viable)
                }
                ParseResult::Error => None,
            }
        }
    }

    #[allow(dead_code)]
    pub fn viable_now(&self) {
        let v = self.byte_states.last().unwrap().viable;
        self.print_viable("now", self.vobset.resolve(v))
    }

    pub fn get_stats(&self) -> String {
        let mut s = self.stats.write().unwrap();
        let r = format!("yacc: {}/{}", s.yacc_actions, s.states_pushed);
        s.yacc_actions = 0;
        s.states_pushed = 0;
        r
    }

    fn mk_byte_state(
        &self,
        ls: LexerState,
        pstack: PStackIdx,
        viable: VobIdx,
    ) -> Option<ByteState> {
        {
            let mut s = self.stats.write().unwrap();
            s.states_pushed += 1;
        }
        if self.vobset.and_is_zero(viable, ls.reachable) {
            None
        } else {
            // print!(
            //     " {:?} {:?} ",
            //     self.vobset.resolve(viable),
            //     self.vobset.resolve(ls.reachable)
            // );
            Some(ByteState {
                lexer_state: ls.state,
                parse_stack_idx: pstack,
                viable,
            })
        }
    }
}

#[derive(Clone, Copy)]
struct PStackIdx(usize);

#[derive(Clone)]
struct ByteState {
    lexer_state: StateID,
    parse_stack_idx: PStackIdx,
    viable: VobIdx,
}

impl Recognizer for CfgParser {
    fn pop_bytes(&mut self, num: usize) {
        self.byte_states.truncate(self.byte_states.len() - num);
    }

    fn collapse(&mut self) {
        let final_state = self.byte_states.pop().unwrap();
        self.byte_states.clear();
        self.byte_states.push(final_state);
    }

    fn special_allowed(&mut self, tok: SpecialToken) -> bool {
        match tok {
            SpecialToken::EndOfSentence => {
                if let Some(st) = self.try_push(None) {
                    let tidx = self.grm.eof_token_idx();
                    let mut pstack = self.pstack_for(&st).clone();
                    matches!(self.parse_lexeme(tidx, &mut pstack), ParseResult::Accept)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn trie_finished(&mut self) {
        assert!(self.byte_states.len() == 1);
    }

    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if let Some(st) = self.try_push(Some(byte)) {
            self.byte_states.push(st);
            true
        } else {
            false
        }
    }
}
