use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use super::{
    device_token_logprobs, DeviceVerification, SpeculativeProposalDistribution,
    VerificationSampler, VerificationSamplingContext,
};
use crate::{
    pipeline::sampling::tests::{
        budget_test_tokenizer, sampled_test_sequence, terminal_test_sequence,
    },
    reasoning_parsers::TagReasoningContext,
    sampler::Logprobs,
    sequence::Sequence,
    speculative::SparseSpeculativeProbs,
};

const A: u32 = 0;
const B: u32 = 1;
const CLOSE: &str = "</think>";
const MODEL_LEN: usize = 1024;

struct Fixture {
    tokenizer: Arc<Tokenizer>,
    rng: Arc<Mutex<Isaac64Rng>>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            tokenizer: budget_test_tokenizer(&["a", "b", "prompt", CLOSE]),
            rng: Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0))),
        }
    }

    fn sequence(&self, limit: usize, stochastic: bool) -> Sequence {
        let mut seq = if stochastic {
            sampled_test_sequence(0, -1, 1.0)
        } else {
            terminal_test_sequence(vec![], None, false)
        };
        seq.enable_reasoning(TagReasoningContext::new_in_think_block());
        seq.set_reasoning_budget(limit, self.tokenizer.clone())
            .unwrap();
        seq
    }

    fn logits(&self, preferred: &[u32]) -> Tensor {
        let vocab = self.tokenizer.get_vocab_size(true);
        let mut values = vec![f32::NEG_INFINITY; preferred.len() * vocab];
        for (row, &token) in preferred.iter().enumerate() {
            values[row * vocab + token as usize] = 0.0;
        }
        Tensor::from_vec(values, (1, preferred.len(), vocab), &Device::Cpu).unwrap()
    }

    fn context<'a>(
        &self,
        logits: &'a Tensor,
        proposal: &'a [u32],
    ) -> VerificationSamplingContext<'a> {
        VerificationSamplingContext {
            logits,
            proposal,
            eos_tok: None,
            llg_factory: None,
            max_model_len: MODEL_LEN,
            rng: self.rng.clone(),
        }
    }

    fn observe(&self, seq: &mut Sequence, token: Logprobs) {
        let bytes = self
            .tokenizer
            .decode(&[token.token], false)
            .unwrap()
            .into_bytes();
        let special = self.tokenizer.token_to_id(CLOSE) == Some(token.token);
        seq.observe_reasoning_token(&bytes, special);
        seq.add_token(token, bytes, None);
    }
}

#[tokio::test]
async fn budget_forces_regular_stochastic_and_device_tokens_including_bonus() {
    let fixture = Fixture::new();
    let close = fixture.tokenizer.token_to_id(CLOSE).unwrap();
    for (device, stochastic, bonus) in [
        (None, false, false),
        (None, true, false),
        (Some(DeviceVerification::TargetTokens(vec![A])), false, true),
    ] {
        let mut seq = fixture.sequence(1, stochastic);
        fixture.observe(&mut seq, device_token_logprobs(A));
        let proposal = if bonus { vec![] } else { vec![A] };
        let distribution =
            stochastic.then(|| SpeculativeProposalDistribution::Logits(fixture.logits(&proposal)));
        let mut sampling =
            VerificationSampler::new(&seq, device, distribution, proposal.len()).unwrap();
        let logits = fixture.logits(&vec![A; proposal.len() + 1]);
        let (token, accepted) = sampling
            .sample(&mut seq, 0, &fixture.context(&logits, &proposal))
            .await
            .unwrap();
        assert_eq!((token.token, accepted, token.logprob), (close, false, 0.0));
        fixture.observe(&mut seq, token);
        assert_eq!(seq.reasoning_tokens(), Some(1));
        assert_eq!(seq.get_reasoning_content().as_deref(), Some("a"));
    }
}

#[tokio::test]
async fn forced_match_overrides_sparse_rejection_and_verifies_the_tail() {
    let fixture = Fixture::new();
    let close = fixture.tokenizer.token_to_id(CLOSE).unwrap();
    for sparse in [false, true] {
        for tail in [A, B] {
            let mut seq = fixture.sequence(0, true);
            let proposal = [close, tail];
            let distribution = sparse.then(|| {
                SpeculativeProposalDistribution::SparseProbs(
                    SparseSpeculativeProbs::new(
                        Tensor::from_vec(proposal.to_vec(), (2, 1), &Device::Cpu).unwrap(),
                        Tensor::ones((2, 1), DType::F32, &Device::Cpu).unwrap(),
                    )
                    .unwrap(),
                )
            });
            let device = DeviceVerification::SparseRejection {
                accepted_drafts: 0,
                continuation_token: A,
            };
            let mut sampling =
                VerificationSampler::new(&seq, Some(device), distribution, proposal.len()).unwrap();
            let logits = fixture.logits(&[A, B, B]);
            let context = fixture.context(&logits, &proposal);
            let (token, accepted) = sampling.sample(&mut seq, 0, &context).await.unwrap();
            assert_eq!((token.token, accepted), (close, true));
            fixture.observe(&mut seq, token);
            let (token, accepted) = sampling.sample(&mut seq, 1, &context).await.unwrap();
            assert_eq!((token.token, accepted), (B, tail == B));
            fixture.observe(&mut seq, token);
            if accepted {
                let (token, accepted) = sampling.sample(&mut seq, 2, &context).await.unwrap();
                assert_eq!((token.token, accepted), (B, false));
            }
            assert_eq!(seq.reasoning_tokens(), Some(0));
            assert_eq!(seq.get_response_content().as_deref(), Some("b"));
        }
    }
}

#[tokio::test]
async fn forced_match_before_device_rejection_preserves_the_device_decision() {
    let fixture = Fixture::new();
    let close = fixture.tokenizer.token_to_id(CLOSE).unwrap();
    let mut seq = fixture.sequence(0, true);
    let proposal = [close, A];
    let device = DeviceVerification::SparseRejection {
        accepted_drafts: proposal.len(),
        continuation_token: B,
    };
    let mut sampling = VerificationSampler::new(&seq, Some(device), None, proposal.len()).unwrap();
    let logits = fixture.logits(&[A, B, B]);
    let context = fixture.context(&logits, &proposal);
    for (idx, expected) in [(0, (close, true)), (1, (A, true)), (2, (B, false))] {
        let (token, accepted) = sampling.sample(&mut seq, idx, &context).await.unwrap();
        assert_eq!((token.token, accepted), expected);
        fixture.observe(&mut seq, token);
    }
    assert_eq!(seq.reasoning_tokens(), Some(0));
    assert_eq!(seq.get_response_content().as_deref(), Some("ab"));
}
