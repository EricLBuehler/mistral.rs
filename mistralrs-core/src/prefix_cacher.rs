use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use candle_core::Result;
use tracing::info;

use crate::{pipeline::KvCache, sequence::Sequence};

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

// A trie node storing optional cache and linked-list pointers for LRU.
type NodeRef = Arc<Mutex<TrieNode>>;
struct TrieNode {
    children: HashMap<u32, NodeRef>,
    cache: Option<CacheElement>,
    prev: Option<Weak<Mutex<TrieNode>>>,
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
            info!("PrefixCacherV2 (RadixAttention) is enabled! Expect higher multi-turn prompt throughput.");
        }
        let root = Arc::new(Mutex::new(TrieNode {
            children: HashMap::new(),
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

    // Move node to the tail of the LRU list
    fn touch_node(&mut self, node: &NodeRef) {
        // Remove from current position
        {
            let nb = node.lock().unwrap();
            if let Some(prev_w) = &nb.prev {
                if let Some(prev) = prev_w.upgrade() {
                    prev.lock().unwrap().next = nb.next.clone();
                }
            } else {
                self.lru_head = nb.next.clone();
            }
            if let Some(next) = &nb.next {
                next.lock().unwrap().prev = nb.prev.clone();
            } else {
                self.lru_tail = nb.prev.as_ref().and_then(|w| w.upgrade());
            }
        }
        // Insert at tail
        if let Some(old_tail) = self.lru_tail.take() {
            old_tail.lock().unwrap().next = Some(Arc::clone(node));
            node.lock().unwrap().prev = Some(Arc::downgrade(&old_tail));
        } else {
            self.lru_head = Some(Arc::clone(node));
            node.lock().unwrap().prev = None;
        }
        node.lock().unwrap().next = None;
        self.lru_tail = Some(Arc::clone(node));
    }

    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache {
            return;
        }
        let cache_data = seq.normal_cache().to_vec();
        let img_hashes = seq.image_hashes().map(|x| x.to_vec());
        let mut node = Arc::clone(&self.root);
        for &tok in seq.get_toks() {
            let next = {
                let mut nb = node.lock().unwrap();
                nb.children
                    .entry(tok)
                    .or_insert_with(|| {
                        Arc::new(Mutex::new(TrieNode {
                            children: HashMap::new(),
                            cache: None,
                            prev: None,
                            next: None,
                        }))
                    })
                    .clone()
            };
            node = next;
        }
        let mut nb = node.lock().unwrap();
        if nb.cache.is_none() {
            self.current_on_device += 1;
        }
        nb.cache = Some(CacheElement {
            cache: cache_data,
            image_hashes: img_hashes,
        });
        drop(nb);
        self.touch_node(&node);
        let _ = self.evict_caches(); // enforce capacity
    }

    pub fn evict_caches(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        let mut evicted = 0;
        while self.current_on_device > self.n_on_device {
            if let Some(head) = self.lru_head.take() {
                let mut hb = head.lock().unwrap();
                if hb.cache.take().is_some() {
                    evicted += 1;
                    self.current_on_device -= 1;
                }
                self.lru_head = hb.next.clone();
                if let Some(new_head) = &self.lru_head {
                    new_head.lock().unwrap().prev = None;
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
        self.root.lock().unwrap().children.clear();
        self.root.lock().unwrap().cache.take();
        self.lru_head = None;
        self.lru_tail = None;
        self.current_on_device = 0;
        Ok(count)
    }

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

        for (i, &tok) in toks.iter().enumerate() {
            // Clone the child reference inside a separate scope so the borrow ends immediately.
            let next_opt = {
                let nb = node.lock().unwrap();
                nb.children.get(&tok).cloned()
            };
            if let Some(child) = next_opt {
                node = child;
                let nb = node.lock().unwrap();
                if nb.cache.is_some() {
                    best = Some(Arc::clone(&node));
                    match_len = i + 1;
                    if let (Some(inp), Some(cached)) =
                        (image_hashes, &nb.cache.as_ref().unwrap().image_hashes)
                    {
                        img_match = inp.iter().zip(cached).take_while(|(a, b)| a == b).count();
                    }
                }
            } else {
                break;
            }
        }

        if let Some(match_node) = best {
            info!("hit!");
            self.touch_node(&match_node);
            let mb = match_node.lock().unwrap();
            let mut elem = mb.cache.clone().unwrap();
            for layer in elem.cache.iter_mut().flatten() {
                layer.set_len(match_len)?;
            }
            let keep_imgs = if let Some(inp) = image_hashes {
                inp.len() - img_match
            } else {
                0
            };
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
