use crate::aici::{
    recognizer::{FunctionalRecognizer, StackRecognizer},
    toktree::SpecialToken,
};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

pub type RecRxState = StateID;

#[derive(Clone)]
pub struct RecRx {
    dfa: dense::DFA<Vec<u32>>,
}

pub type RxStackRecognizer = StackRecognizer<StateID, RecRx>;

impl RecRx {
    pub fn from_rx(rx: &str) -> Self {
        let rx = if rx.ends_with("$") {
            rx.to_string()
        } else {
            rx.to_string() + "$"
        };
        let dfa = dense::Builder::new()
            .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build(&rx)
            .unwrap();
        println!("dfa: {} bytes", dfa.memory_usage());
        Self { dfa }
    }

    pub fn to_stack_recognizer(self) -> RxStackRecognizer {
        StackRecognizer::from(self)
    }
}

impl FunctionalRecognizer<RecRxState> for RecRx {
    fn initial(&self) -> RecRxState {
        self.dfa
            .universal_start_state(regex_automata::Anchored::Yes)
            .expect("dfa has no universal start state; make sure it doesn't match empty string")
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
