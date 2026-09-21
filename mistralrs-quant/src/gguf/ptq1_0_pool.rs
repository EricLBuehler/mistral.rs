//! Spin-then-park worker pool for the small decode matmuls, where rayon's workers sleep between layers.

use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread::{self, Thread},
    time::{Duration, Instant},
};

const DEFAULT_SPIN: Duration = Duration::from_micros(20);
const SPINS_PER_CLOCK_CHECK: u32 = 64;

struct Job {
    tasks: usize,
    run: *const (dyn Fn(usize) + Sync),
}

struct Shared {
    generation: AtomicU64,
    job: AtomicPtr<Job>,
    next: AtomicUsize,
    done: AtomicUsize,
    active: AtomicUsize,
    parked: Vec<AtomicBool>,
    threads: OnceLock<Vec<Thread>>,
    spin: Duration,
}

pub(super) struct Pool {
    shared: Arc<Shared>,
    caller: Mutex<()>,
}

// SAFETY: `Job::run` is only dereferenced while `Pool::run` keeps the closure alive (see `Pool::run`)
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// A `&mut` slice that tasks split into disjoint chunks by index.
pub(super) struct Chunks<T> {
    ptr: UnsafeCell<*mut T>,
    len: usize,
    chunk: usize,
}

unsafe impl<T: Send> Sync for Chunks<T> {}

impl<T> Chunks<T> {
    pub(super) fn new(data: &mut [T], chunk: usize) -> Self {
        Self {
            ptr: UnsafeCell::new(data.as_mut_ptr()),
            len: data.len(),
            chunk,
        }
    }

    /// # Safety
    /// Each `index` must be taken by at most one task at a time.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn get(&self, index: usize) -> &mut [T] {
        let start = index * self.chunk;
        let end = (start + self.chunk).min(self.len);
        std::slice::from_raw_parts_mut((*self.ptr.get()).add(start), end - start)
    }
}

fn work(shared: &Shared, job: &Job) {
    loop {
        let i = shared.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.tasks {
            return;
        }
        // SAFETY: the caller of `Pool::run` waits for `active == 0` before dropping the closure
        unsafe { (*job.run)(i) };
        shared.done.fetch_add(1, Ordering::Release);
    }
}

fn worker(shared: Arc<Shared>, id: usize) {
    let mut seen = 0;
    let mut since = Instant::now();
    let mut spins = 0u32;
    loop {
        let generation = shared.generation.load(Ordering::Acquire);
        if generation == seen {
            spins += 1;
            if spins < SPINS_PER_CLOCK_CHECK {
                std::hint::spin_loop();
                continue;
            }
            spins = 0;
            if since.elapsed() < shared.spin {
                continue;
            }
            shared.parked[id].store(true, Ordering::SeqCst);
            if shared.generation.load(Ordering::SeqCst) == seen {
                thread::park();
            }
            shared.parked[id].store(false, Ordering::SeqCst);
            since = Instant::now();
            continue;
        }
        seen = generation;
        shared.active.fetch_add(1, Ordering::SeqCst);
        let job = shared.job.load(Ordering::SeqCst);
        if !job.is_null() {
            // SAFETY: `active` was raised before the load, so `Pool::run` cannot return while we hold this
            work(&shared, unsafe { &*job });
        }
        shared.active.fetch_sub(1, Ordering::SeqCst);
        since = Instant::now();
    }
}

impl Pool {
    fn new() -> Self {
        let logical = thread::available_parallelism().map_or(1, |n| n.get());
        let workers = logical - 1;
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            parked: (0..workers).map(|_| AtomicBool::new(false)).collect(),
            threads: OnceLock::new(),
            spin: DEFAULT_SPIN,
        });
        let handles = (0..workers)
            .map(|id| {
                let shared = shared.clone();
                thread::Builder::new()
                    .name(format!("ptq1-0-{id}"))
                    .spawn(move || worker(shared, id))
                    .expect("spawn PTQ1_0 worker")
                    .thread()
                    .clone()
            })
            .collect();
        let _ = shared.threads.set(handles);
        Self {
            shared,
            caller: Mutex::new(()),
        }
    }

    pub(super) fn global() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(Pool::new)
    }

    /// Runs `f(0..tasks)` across the workers and the calling thread, returning when every task is done.
    pub(super) fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        let shared = &*self.shared;
        if tasks <= 1 || shared.parked.is_empty() {
            (0..tasks).for_each(f);
            return;
        }
        let _one_caller = self.caller.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: erased lifetime; the closure outlives every use because we wait for `done` and `active` below
        let run: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Job { tasks, run };
        shared.next.store(0, Ordering::Relaxed);
        shared.done.store(0, Ordering::Relaxed);
        shared
            .job
            .store(&job as *const Job as *mut Job, Ordering::SeqCst);
        shared.generation.fetch_add(1, Ordering::SeqCst);
        let threads = shared
            .threads
            .get()
            .expect("workers registered at construction");
        for (parked, thread) in shared.parked.iter().zip(threads) {
            if parked.load(Ordering::SeqCst) {
                thread.unpark();
            }
        }
        work(shared, &job);
        while shared.done.load(Ordering::Acquire) < tasks {
            std::hint::spin_loop();
        }
        shared.job.store(std::ptr::null_mut(), Ordering::SeqCst);
        while shared.active.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use super::*;

    #[test]
    fn every_task_runs_exactly_once() {
        let pool = Pool::global();
        for round in 0..2000usize {
            let tasks = 1 + round % 97;
            let hits: Vec<AtomicU64> = (0..tasks).map(|_| AtomicU64::new(0)).collect();
            pool.run(tasks, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn disjoint_chunks_are_all_written() {
        let pool = Pool::global();
        let mut data = vec![0u32; 1000];
        let chunks = Chunks::new(&mut data, 16);
        pool.run(data.len().div_ceil(16), &|i| {
            // SAFETY: task `i` is the only one touching chunk `i`
            let c = unsafe { chunks.get(i) };
            c.iter_mut().for_each(|v| *v = i as u32 + 1);
        });
        assert!(data
            .iter()
            .enumerate()
            .all(|(j, v)| *v == (j / 16) as u32 + 1));
    }

    #[test]
    fn survives_idle_gaps_that_park_the_workers() {
        let pool = Pool::global();
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(3));
            let sum = AtomicU64::new(0);
            pool.run(64, &|i| {
                sum.fetch_add(i as u64, Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), (0..64).sum::<u64>());
        }
    }
}
