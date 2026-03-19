#![cfg(test)]

use super::*;
use std::sync::Arc;
use std::thread;

#[test]
fn test_rwlock_basic() {
    let lock = RwLock::new(0);
    *lock.write() = 42;
    assert_eq!(*lock.read(), 42);
}

#[test]
fn test_rwlock_concurrent_reads() {
    let lock = Arc::new(RwLock::new(0));
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let l = lock.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let _r = l.read();
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
fn test_rwlock_concurrent_writes() {
    let lock = Arc::new(RwLock::new(0));
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let l = lock.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    *l.write() += 1;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(*lock.read(), 1000);
}

#[test]
fn test_response_cache_store_retrieve() {
    let cache = InMemoryResponseCache::new();
    let response = ResponsesObject {
        id: "test-id".to_string(),
        object: "response",
        created_at: 1234567890.0,
        model: "test-model".to_string(),
        status: "completed".to_string(),
        output: vec![],
        output_text: None,
        usage: None,
        error: None,
        metadata: None,
        instructions: None,
        incomplete_details: None,
    };

    cache
        .store_response("test-id".to_string(), response.clone())
        .unwrap();
    let retrieved = cache.get_response("test-id").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, "test-id");
}

#[test]
fn test_response_cache_delete() {
    let cache = InMemoryResponseCache::new();
    let response = ResponsesObject {
        id: "test-id".to_string(),
        object: "response",
        created_at: 1234567890.0,
        model: "test-model".to_string(),
        status: "completed".to_string(),
        output: vec![],
        output_text: None,
        usage: None,
        error: None,
        metadata: None,
        instructions: None,
        incomplete_details: None,
    };

    cache
        .store_response("test-id".to_string(), response.clone())
        .unwrap();
    let deleted = cache.delete_response("test-id").unwrap();
    assert!(deleted);
    let retrieved = cache.get_response("test-id").unwrap();
    assert!(retrieved.is_none());
}

#[test]
fn test_response_cache_concurrent_access() {
    let cache = Arc::new(InMemoryResponseCache::new());
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let c = cache.clone();
            thread::spawn(move || {
                let response = ResponsesObject {
                    id: format!("test-id-{}", i),
                    object: "response",
                    created_at: 1234567890.0,
                    model: "test-model".to_string(),
                    status: "completed".to_string(),
                    output: vec![],
                    output_text: None,
                    usage: None,
                    error: None,
                    metadata: None,
                    instructions: None,
                    incomplete_details: None,
                };
                c.store_response(format!("test-id-{}", i), response)
                    .unwrap();
                let _ = c.get_response(&format!("test-id-{}", i)).unwrap();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn test_lock_ordering_preserved() {
    // This test verifies that the lock ordering (responses -> chunks -> histories) is preserved
    let cache = Arc::new(InMemoryResponseCache::new());
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let c = cache.clone();
            thread::spawn(move || {
                for j in 0..50 {
                    let id = format!("test-{}-{}", i, j);
                    let response = ResponsesObject {
                        id: id.clone(),
                        object: "response",
                        created_at: 1234567890.0,
                        model: "test-model".to_string(),
                        status: "completed".to_string(),
                        output: vec![],
                        output_text: None,
                        usage: None,
                        error: None,
                        metadata: None,
                        instructions: None,
                        incomplete_details: None,
                    };
                    // Store response
                    c.store_response(id.clone(), response).unwrap();
                    // Store chunks
                    c.store_chunks(id.clone(), vec![]).unwrap();
                    // Store history
                    c.store_conversation_history(id.clone(), vec![]).unwrap();
                    // Delete (tests lock ordering in delete_response)
                    c.delete_response(&id).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
