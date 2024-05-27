use crate::aici::{recognizer::FunctionalRecognizer, toktree::SpecialToken};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

pub type RecRxState = StateID;

#[derive(Clone)]
pub struct RecRx {
    dfa: dense::DFA<Vec<u32>>,
}

impl RecRx {
    pub fn from_rx(rx: &str) -> anyhow::Result<Self> {
        let rx = if rx.ends_with('$') {
            rx.to_string()
        } else {
            rx.to_string() + "$"
        };
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build(&rx)
            .map_err(|e| anyhow::Error::msg(format!("Could not compile regex `{rx}` - {e:?}")))?;

        Ok(Self { dfa })
    }
}

impl FunctionalRecognizer<RecRxState> for RecRx {
    fn initial(&self) -> anyhow::Result<RecRxState> {
        let state = self
            .dfa
            .universal_start_state(regex_automata::Anchored::Yes);

        match state {
            Some(state) => Ok(state),
            None => {
                let err = anyhow::Error::msg(
                    "dfa has no universal start state; make sure it doesn't match empty string",
                );
                Err(err)
            }
        }
    }

    #[inline(always)]
    fn append(&self, state: RecRxState, byte: u8) -> RecRxState {
        self.dfa.next_state(state, byte)
    }

    #[inline(always)]
    fn byte_allowed(&self, state: RecRxState, byte: u8) -> bool {
        !self.dfa.is_dead_state(self.dfa.next_state(state, byte))
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
