use ahash::AHasher;
use candle_core::Result;
use hashbrown::HashMap;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::{
    hash::BuildHasherDefault,
    sync::{Arc, Weak},
};
use tracing::info;

use crate::{pipeline::KvCache, sequence::Sequence};

type AHashMap<K, V> = HashMap<K, V, BuildHasherDefault<AHasher>>;
type NodeRef = Arc<RwLock<TrieNode>>;

#[derive(Clone)]
struct CacheElement {
    cache: Vec<Option<KvCache>>,
    image_hashes: Option<Vec<u64>>,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: Vec<Option<KvCache>>,
    pub images_to_keep: usize,
    pub toks: Vec<u32>,
    pub offset: usize,
}

struct TrieNode {
    children: AHashMap<u32, NodeRef>,
    cache: Option<CacheElement>,
    prev: Option<Weak<RwLock<TrieNode>>>,
    next: Option<NodeRef>,
}

pub struct PrefixCacheManagerV2 {
    root: NodeRef,
    n_on_device: usize,
    no_prefix_cache: bool,
    lru_head: Option<NodeRef>,
    lru_tail: Option<NodeRef>,
    current_on_device: usize,
}

impl PrefixCacheManagerV2 {
    pub fn new(n_on_device: usize, no_prefix_cache: bool) -> Self {
        if !no_prefix_cache {
            info!("PrefixCacherV2 (RadixAttention) enabled - expect higher multi-turn throughput.");
        }
        // Pre‑seed our hash map with a small capacity (common branching factor)
        let root = Arc::new(RwLock::new(TrieNode {
            children: AHashMap::with_capacity_and_hasher(8, Default::default()),
            cache: None,
            prev: None,
            next: None,
        }));
        PrefixCacheManagerV2 {
            root,
            n_on_device,
            no_prefix_cache,
            lru_head: None,
            lru_tail: None,
            current_on_device: 0,
        }
    }

    /// Bump a node to the most‑recently‑used tail
    fn touch_node(&mut self, node: &NodeRef) {
        // Take a write lock to unlink & relink
        let mut nb: RwLockWriteGuard<'_, TrieNode> = node.write();
        // Unlink from list
        if let Some(prev_w) = nb.prev.take() {
            if let Some(prev) = prev_w.upgrade() {
                let mut prev_nb = prev.write();
                prev_nb.next = nb.next.clone();
            }
        } else {
            // was head
            self.lru_head = nb.next.clone();
        }
        if let Some(next) = nb.next.take() {
            let mut next_nb = next.write();
            next_nb.prev = nb.prev.clone();
        } else {
            // was tail
            self.lru_tail = nb.prev.as_ref().and_then(|w| w.upgrade());
        }
        drop(nb);

        // Link at tail
        if let Some(old_tail) = self.lru_tail.take() {
            {
                let mut tail_nb = old_tail.write();
                tail_nb.next = Some(Arc::clone(node));
            }
            let mut me_nb = node.write();
            me_nb.prev = Some(Arc::downgrade(&old_tail));
            me_nb.next = None;
        } else {
            // empty list
            self.lru_head = Some(Arc::clone(node));
            let mut me_nb = node.write();
            me_nb.prev = None;
            me_nb.next = None;
        }
        self.lru_tail = Some(Arc::clone(node));
    }

    /// Insert or update the cache for a full token sequence
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache {
            return;
        }
        let data = seq.normal_cache().to_vec();
        let img_hashes = seq.image_hashes().map(|v| v.to_vec());
        let mut node = Arc::clone(&self.root);

        // Walk/build trie
        for &tok in seq.get_toks() {
            let next = {
                let mut nb = node.write();
                nb.children
                    .entry(tok)
                    .or_insert_with(|| {
                        Arc::new(RwLock::new(TrieNode {
                            children: AHashMap::with_capacity_and_hasher(8, Default::default()),
                            cache: None,
                            prev: None,
                            next: None,
                        }))
                    })
                    .clone()
            };
            node = next;
        }

        {
            let mut nb = node.write();
            if nb.cache.is_none() {
                self.current_on_device += 1;
            }
            nb.cache = Some(CacheElement {
                cache: data,
                image_hashes: img_hashes,
            });
        }

        self.touch_node(&node);
        // Enforce capacity
        let _ = self.evict_caches();
    }

    /// Evict least‑recently used entries until under capacity
    pub fn evict_caches(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        let mut evicted = 0;
        while self.current_on_device > self.n_on_device {
            if let Some(head) = self.lru_head.take() {
                {
                    let mut hb = head.write();
                    if hb.cache.take().is_some() {
                        evicted += 1;
                        self.current_on_device -= 1;
                    }
                    // unlink head
                    self.lru_head = hb.next.clone();
                }
                if let Some(new_head) = &self.lru_head {
                    let mut hnb = new_head.write();
                    hnb.prev = None;
                } else {
                    self.lru_tail = None;
                }
            } else {
                break;
            }
        }
        Ok(evicted)
    }

    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let count = self.current_on_device;
        {
            let mut root_nb = self.root.write();
            root_nb.children.clear();
            root_nb.cache = None;
        }
        self.lru_head = None;
        self.lru_tail = None;
        self.current_on_device = 0;
        Ok(count)
    }

    /// Lookup the longest‐matching prefix cache for `toks`
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        _contains_images: bool,
    ) -> Result<Option<MatchingCache>> {
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        let mut node = Arc::clone(&self.root);
        let mut best: Option<NodeRef> = None;
        let mut match_len = 0;
        let mut img_match = 0;

        // Fast, shared‐read walk
        for (i, &tok) in toks.iter().enumerate() {
            // Acquire *shared* read lock just to access children map
            let next_opt = {
                let rb: RwLockReadGuard<'_, TrieNode> = node.read();
                rb.children.get(&tok).cloned()
            };
            if let Some(child) = next_opt {
                node = child;
                let rb = node.read();
                if rb.cache.is_some() {
                    best = Some(Arc::clone(&node));
                    match_len = i + 1;
                    if let (Some(inp), Some(cached)) =
                        (image_hashes, &rb.cache.as_ref().unwrap().image_hashes)
                    {
                        img_match = inp.iter().zip(cached).take_while(|(a, b)| a == b).count();
                    }
                }
            } else {
                break;
            }
        }

        if let Some(matched_node) = best {
            // Bump in LRU
            self.touch_node(&matched_node);

            // Prepare result
            let rb = matched_node.read();
            let mut elem = rb.cache.clone().unwrap();
            for layer in elem.cache.iter_mut().flatten() {
                layer.set_len(match_len)?;
            }
            let keep_imgs = image_hashes.map_or(0, |inp| inp.len().saturating_sub(img_match));

            Ok(Some(MatchingCache {
                normal: elem.cache,
                images_to_keep: keep_imgs,
                toks: toks[match_len..].to_vec(),
                offset: match_len,
            }))
        } else {
            Ok(None)
        }
    }
}
