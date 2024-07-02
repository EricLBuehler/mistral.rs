use std::error::Error;

use crate::aici::{
    recognizer::{FunctionalRecognizer, StackRecognizer},
    toktree::SpecialToken,
};
use anyhow::{bail, Result};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

pub type RecRxState = StateID;

#[derive(Clone)]
pub struct RecRx {
    dfa: dense::DFA<Vec<u32>>,
    info: String,
}

pub type RxStackRecognizer = StackRecognizer<StateID, RecRx>;

impl RecRx {
    pub fn from_rx(rx: &str, size_limit: Option<usize>) -> Result<Self> {
        let rx = if rx.ends_with('$') {
            rx.to_string()
        } else {
            rx.to_string() + "$"
        };
        let rx = if let Some(stripped) = rx.strip_prefix('^') {
            stripped.to_string()
        } else {
            rx
        };
        // default to 16MB - it takes about 1s to build
        let size_limit = size_limit.unwrap_or(16 << 20);
        let t0 = std::time::Instant::now();
        let cfg = dense::Config::new()
            .start_kind(regex_automata::dfa::StartKind::Anchored)
            .dfa_size_limit(Some(size_limit))
            .determinize_size_limit(Some(size_limit));
        let dfa = dense::Builder::new()
            .configure(cfg)
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build(&rx);
        let dfa = match dfa {
            Ok(dfa) => dfa,
            Err(e) => {
                if let Some(e) = e.source() {
                    if let Some(e) = e.source() {
                        bail!("error building dfa(2): {}", e)
                    } else {
                        bail!("error building dfa(1): {}", e)
                    }
                } else {
                    bail!("error building dfa(0): {}", e)
                }
            }
        };
        let time = t0.elapsed();
        let mb_per_s = dfa.memory_usage() as f64 / time.as_secs_f64() / 1024.0 / 1024.0;
        let info = format!(
            "dfa: {} bytes; time {:?}; {:.3} MB/s",
            dfa.memory_usage(),
            time,
            mb_per_s
        );

        if let Err(e) = dfa.start_state(&anchored_start()) {
            bail!("DFA has no start state; {}", e)
        }

        Ok(Self { dfa, info })
    }

    pub fn info(&self) -> &str {
        &self.info
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_stack_recognizer(self) -> RxStackRecognizer {
        StackRecognizer::from(self)
    }
}

fn anchored_start() -> regex_automata::util::start::Config {
    regex_automata::util::start::Config::new().anchored(regex_automata::Anchored::Yes)
}

impl FunctionalRecognizer<RecRxState> for RecRx {
    fn initial(&self) -> RecRxState {
        self.dfa
            .start_state(&anchored_start())
            .expect("dfa has no start state")
    }

    #[inline(always)]
    fn try_append(&self, state: RecRxState, byte: u8) -> Option<RecRxState> {
        let next = self.dfa.next_state(state, byte);
        if self.dfa.is_dead_state(next) {
            None
        } else {
            Some(next)
        }
    }

    #[inline(always)]
    fn special_allowed(&self, state: RecRxState, tok: SpecialToken) -> bool {
        let state = self.dfa.next_eoi_state(state);
        match tok {
            SpecialToken::EndOfSentence => self.dfa.is_match_state(state),
            _ => false,
        }
    }
}
