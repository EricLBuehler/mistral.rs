//! Benchmarks for lock performance under various contention scenarios

use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Benchmark mutex operations with varying thread counts
pub fn bench_mutex_contention(num_threads: usize, iterations: usize) -> Duration {
    let mutex = Arc::new(Mutex::new(0u64));
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let m = mutex.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    *m.lock() += 1;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    start.elapsed()
}

/// Benchmark RwLock read operations
pub fn bench_rwlock_reads(num_threads: usize, iterations: usize) -> Duration {
    let lock = Arc::new(RwLock::new(42u64));
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let l = lock.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    let _val = *l.read();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    start.elapsed()
}

/// Benchmark RwLock write operations
pub fn bench_rwlock_writes(num_threads: usize, iterations: usize) -> Duration {
    let lock = Arc::new(RwLock::new(0u64));
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let l = lock.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    *l.write() += 1;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    start.elapsed()
}

/// Benchmark mixed read/write workload (90% reads, 10% writes)
pub fn bench_rwlock_mixed(num_threads: usize, iterations: usize) -> Duration {
    let lock = Arc::new(RwLock::new(0u64));
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let l = lock.clone();
            thread::spawn(move || {
                for i in 0..iterations {
                    // 10% writes, 90% reads
                    if (thread_id + i) % 10 == 0 {
                        *l.write() += 1;
                    } else {
                        let _val = *l.read();
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutex_low_contention() {
        let duration = bench_mutex_contention(2, 1000);
        println!("Low contention (2 threads): {:?}", duration);
    }

    #[test]
    fn test_mutex_medium_contention() {
        let duration = bench_mutex_contention(8, 1000);
        println!("Medium contention (8 threads): {:?}", duration);
    }

    #[test]
    fn test_mutex_high_contention() {
        let duration = bench_mutex_contention(32, 1000);
        println!("High contention (32 threads): {:?}", duration);
    }

    #[test]
    fn test_rwlock_read_performance() {
        let duration = bench_rwlock_reads(8, 1000);
        println!("RwLock reads (8 threads): {:?}", duration);
    }

    #[test]
    fn test_rwlock_write_performance() {
        let duration = bench_rwlock_writes(8, 1000);
        println!("RwLock writes (8 threads): {:?}", duration);
    }

    #[test]
    fn test_rwlock_mixed_workload() {
        let duration = bench_rwlock_mixed(8, 1000);
        println!("RwLock mixed workload (8 threads): {:?}", duration);
    }

    #[test]
    fn bench_comparison() {
        println!("\n=== Lock Performance Benchmarks ===\n");

        println!("Mutex Contention:");
        for threads in [2, 8, 16, 32] {
            let duration = bench_mutex_contention(threads, 1000);
            println!("  {} threads: {:?}", threads, duration);
        }

        println!("\nRwLock Read Performance:");
        for threads in [2, 8, 16, 32] {
            let duration = bench_rwlock_reads(threads, 1000);
            println!("  {} threads: {:?}", threads, duration);
        }

        println!("\nRwLock Write Performance:");
        for threads in [2, 8, 16, 32] {
            let duration = bench_rwlock_writes(threads, 1000);
            println!("  {} threads: {:?}", threads, duration);
        }

        println!("\nRwLock Mixed Workload (90% reads, 10% writes):");
        for threads in [2, 8, 16, 32] {
            let duration = bench_rwlock_mixed(threads, 1000);
            println!("  {} threads: {:?}", threads, duration);
        }
    }
}
