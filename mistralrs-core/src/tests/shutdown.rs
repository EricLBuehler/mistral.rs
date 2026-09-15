use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Mutex, RwLock,
    },
    thread,
    time::Duration,
};

use crate::{
    shutdown::{engine_runtime, join_engines, EngineThreadScope, RetiringEngine, Shutdown},
    MistralRs, MistralRsError, MistralRsShutdownFailure, Request,
};

const DEADLINE: Duration = Duration::from_secs(5);

struct Resources {
    drops: Arc<AtomicUsize>,
    panic: bool,
}

impl Drop for Resources {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
        assert!(!self.panic, "controlled resource release failure");
    }
}

fn worker(
    drops: Arc<AtomicUsize>,
    panic_after_signal: bool,
    panic_resources: bool,
) -> (
    RetiringEngine<Resources>,
    mpsc::Receiver<()>,
    mpsc::Sender<()>,
) {
    let (sender, mut requests) = tokio::sync::mpsc::channel(1);
    let (terminated, observed) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let worker = thread::spawn(move || {
        let _scope = EngineThreadScope::enter();
        assert!(matches!(requests.blocking_recv(), Some(Request::Terminate)));
        terminated.send(()).expect("termination observer");
        released
            .recv_timeout(DEADLINE)
            .expect("release native worker");
        assert!(!panic_after_signal, "controlled native worker failure");
    });
    (
        RetiringEngine::new(
            sender,
            Some(worker),
            Resources {
                drops,
                panic: panic_resources,
            },
        ),
        observed,
        release,
    )
}

#[test]
fn close_keeps_resources_until_the_actual_worker_thread_exits() {
    let shutdown = Arc::new(Shutdown::default());
    let drops = Arc::new(AtomicUsize::new(0));
    let (engine, terminated, release) = worker(drops.clone(), false, false);
    let (done, completed) = mpsc::channel();
    let first = {
        let shutdown = shutdown.clone();
        thread::spawn(move || {
            let result = shutdown.run(|| join_engines(vec![engine]));
            done.send(()).expect("shutdown observer");
            result
        })
    };
    terminated
        .recv_timeout(DEADLINE)
        .expect("native termination was requested");
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(matches!(
        completed.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    let second = {
        let shutdown = shutdown.clone();
        thread::spawn(move || shutdown.run(|| panic!("shutdown cannot run a second close")))
    };
    release.send(()).expect("release native worker");
    assert!(first.join().expect("first waiter").is_ok());
    assert!(second.join().expect("second waiter").is_ok());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(shutdown
        .run(|| panic!("shutdown was already joined"))
        .is_ok());
}

#[test]
fn a_panicked_engine_does_not_skip_later_joins_and_the_result_is_retained() {
    let shutdown = Arc::new(Shutdown::default());
    let drops = Arc::new(AtomicUsize::new(0));
    let (first, first_terminated, first_release) = worker(drops.clone(), true, false);
    let (second, second_terminated, second_release) = worker(drops.clone(), false, false);
    let waiter = {
        let shutdown = shutdown.clone();
        thread::spawn(move || shutdown.run(|| join_engines(vec![first, second])))
    };
    first_terminated
        .recv_timeout(DEADLINE)
        .expect("first engine signalled");
    second_terminated
        .recv_timeout(DEADLINE)
        .expect("all engines signalled before joining");
    first_release.send(()).expect("release first engine");
    second_release.send(()).expect("release second engine");
    let first_error = waiter
        .join()
        .expect("shutdown owner")
        .expect_err("engine panic reported");
    assert!(first_error
        .failures()
        .contains(&MistralRsShutdownFailure::EnginePanicked));
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    let second_error = shutdown
        .run(|| panic!("cached result must be reused"))
        .expect_err("retained failure");
    assert!(std::ptr::eq(
        first_error.failures(),
        second_error.failures()
    ));
}

#[test]
fn a_resource_destructor_panic_still_releases_other_joined_engines() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (first, first_terminated, first_release) = worker(drops.clone(), false, true);
    let (second, second_terminated, second_release) = worker(drops.clone(), false, false);
    let waiter = thread::spawn(move || join_engines(vec![first, second]));
    first_terminated
        .recv_timeout(DEADLINE)
        .expect("first engine signalled");
    second_terminated
        .recv_timeout(DEADLINE)
        .expect("second engine signalled");
    first_release.send(()).expect("release first engine");
    second_release.send(()).expect("release second engine");
    assert_eq!(
        waiter.join().expect("shutdown owner"),
        vec![MistralRsShutdownFailure::ResourcesPanicked]
    );
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn shutdown_fences_new_calls_and_waits_for_already_admitted_management() {
    let shutdown = Arc::new(Shutdown::default());
    let admission = shutdown.enter().expect("initial admission");
    let (started, observed) = mpsc::channel();
    let waiter = {
        let shutdown = shutdown.clone();
        thread::spawn(move || {
            shutdown.run(|| {
                started.send(()).expect("close observer");
                Vec::new()
            })
        })
    };
    assert!(shutdown.wait_for_shutdown(DEADLINE));
    assert!(matches!(
        shutdown.enter(),
        Err(MistralRsError::ShuttingDown)
    ));
    assert!(matches!(
        observed.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(admission);
    observed
        .recv_timeout(DEADLINE)
        .expect("close starts after admission drains");
    assert!(waiter.join().expect("shutdown owner").is_ok());
}

#[test]
fn engine_threads_cannot_join_their_own_shutdown() {
    let shutdown = Shutdown::default();
    {
        let _scope = EngineThreadScope::enter();
        let failure = shutdown
            .run(|| panic!("engine cannot start its own join"))
            .expect_err("self join rejected");
        assert_eq!(
            failure.failures(),
            &[MistralRsShutdownFailure::EngineThreadCannotJoin]
        );
        assert!(matches!(shutdown.enter(), Err(MistralRsError::Shutdown(_))));
    }
    assert!(shutdown.run(Vec::new).is_ok());
}

#[test]
fn native_runtime_workers_also_reject_self_join() {
    let runtime = engine_runtime().expect("native runtime");
    let asynchronous = runtime.spawn(async {
        Shutdown::default().run(|| panic!("native async worker cannot start shutdown"))
    });
    let blocking = runtime.spawn_blocking(|| {
        Shutdown::default().run(|| panic!("native blocking worker cannot start shutdown"))
    });
    let async_error = runtime
        .block_on(asynchronous)
        .expect("async worker")
        .expect_err("self join rejected");
    let blocking_error = runtime
        .block_on(blocking)
        .expect("blocking worker")
        .expect_err("self join rejected");
    assert_eq!(
        async_error.failures(),
        &[MistralRsShutdownFailure::EngineThreadCannotJoin]
    );
    assert_eq!(
        blocking_error.failures(),
        &[MistralRsShutdownFailure::EngineThreadCannotJoin]
    );
}

#[test]
fn a_management_retirement_failure_fences_replacement_engines() {
    let shutdown = Shutdown::default();
    shutdown
        .record(vec![MistralRsShutdownFailure::EnginePanicked])
        .expect_err("retain failed retirement");
    assert!(matches!(shutdown.enter(), Err(MistralRsError::Shutdown(_))));
    let failure = shutdown
        .run(Vec::new)
        .expect_err("shutdown includes earlier native failure");
    assert_eq!(
        failure.failures(),
        &[MistralRsShutdownFailure::EnginePanicked]
    );
}

#[test]
fn model_operation_reservation_is_atomic_and_released_on_early_return() {
    let models = MistralRs {
        shutdown: Shutdown::default(),
        engines: RwLock::new(HashMap::new()),
        unloaded_models: RwLock::new(HashMap::new()),
        reloading_models: RwLock::new(HashSet::new()),
        default_engine_id: RwLock::new(None),
        model_aliases: RwLock::new(HashMap::new()),
        log: None,
        id: "test".to_string(),
        creation_time: 0,
        next_request_id: Mutex::new(std::cell::RefCell::new(0)),
    };
    let operation = models
        .begin_model_operation("test")
        .expect("reserve operation");
    assert!(matches!(
        models.begin_model_operation("test"),
        Err(MistralRsError::ModelReloading(_))
    ));
    drop(operation);
    assert!(models.begin_model_operation("test").is_ok());
    assert!(models.shutdown_blocking().is_ok());
}
