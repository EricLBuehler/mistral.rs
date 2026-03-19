#![cfg(test)]

use super::*;
use std::sync::Arc;
use std::thread;

#[test]
fn test_mutex_basic() {
    let mutex = Mutex::new(0);
    *mutex.lock() = 42;
    assert_eq!(*mutex.lock(), 42);
}

#[test]
fn test_mutex_concurrent() {
    let mutex = Arc::new(Mutex::new(0));
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let m = mutex.clone();
            thread::spawn(move || {
                for _ in 0..1000 {
                    *m.lock() += 1;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(*mutex.lock(), 100_000);
}

#[test]
fn test_normal_cache_construction() {
    let cache = NormalCache::new(4, 512);
    let locked = cache.lock();
    assert_eq!(locked.0.len(), 4);
}

#[test]
fn test_cache_construction() {
    let cache = Cache::new(4, false);
    assert!(!cache.is_xlora());
    
    let cache_xlora = Cache::new(4, true);
    assert!(cache_xlora.is_xlora());
}

#[test]
fn test_cache_lock_access() {
    let cache = Cache::new(4, false);
    let mut locked = cache.lock();
    assert_eq!(locked.len(), 4);
    locked.push(None);
    assert_eq!(locked.len(), 5);
}

#[test]
fn test_concurrent_cache_access() {
    let cache = Arc::new(Cache::new(4, false));
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let c = cache.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let _locked = c.lock();
                    // Simulate some work
                    thread::yield_now();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_normal_cache_sliding() {
    let cache = NormalCache::new_sliding(4, 512, Some(256));
    let locked = cache.lock();
    assert_eq!(locked.0.len(), 4);
    // Verify all caches are rotating type
    for kv_cache in &locked.0 {
        assert!(kv_cache.is_rotating());
    }
}

#[test]
fn test_cache_types() {
    let types = vec![
        NormalCacheType::Normal { max_seq_len: 512 },
        NormalCacheType::SlidingWindow { window: 256 },
    ];
    let cache = NormalCache::from_types(types);
    let locked = cache.lock();
    assert_eq!(locked.0.len(), 2);
    assert!(!locked.0[0].is_rotating());
    assert!(locked.0[1].is_rotating());
}
