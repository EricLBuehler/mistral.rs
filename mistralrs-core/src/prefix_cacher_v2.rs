use candle_core::{Device, Result};
use radix_trie::{Trie, TrieCommon, TrieKey};

use crate::{pipeline::KvCache, sequence::Sequence};

#[derive(PartialEq, Eq, Debug)]
struct Tokens(Vec<u32>);

impl TrieKey for Tokens {
    fn encode_bytes(&self) -> Vec<u8> {
        self.0
            .iter()
            .flat_map(|x| bytemuck::bytes_of(x).to_vec())
            .collect::<Vec<u8>>()
    }
}

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

pub struct PrefixCacheManagerV2 {
    caches: Trie<Tokens, Vec<Option<KvCache>>>,
    device: Device,
    pub n_on_device: usize,
    no_prefix_cache: bool,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: Vec<Option<KvCache>>,
    pub toks: Vec<u32>,
}

impl PrefixCacheManagerV2 {
    pub fn new(device: Device, n_on_device: usize, no_prefix_cache: bool) -> Self {
        PrefixCacheManagerV2 {
            caches: Trie::new(),
            device,
            n_on_device,
            no_prefix_cache,
        }
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache {
            return;
        }
        let cache = seq
            .normal_cache()
            .to_vec();
        self.caches.insert(seq.get_toks().to_vec().into(), cache);
    }

    /// Search for a matching cache given some toks
    pub fn search_for_matching_cache(&mut self, toks: &[u32]) -> Result<Option<MatchingCache>> {
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());
        for (k, v) in self.caches.iter() {
            if &toks.0[0..k.0.len()] == &k.0 {
                println!("found a match! {}", k.0.len());
                return Ok(Some(MatchingCache {
                    normal: v.clone(),
                    toks: toks.0[k.0.len()..].to_vec(),
                }));
            }
        }
        
        Ok(None)
    }
}
