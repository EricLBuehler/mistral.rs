use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    thread,
    time::Duration,
};

use crate::{
    shutdown::{engine_runtime, join_engines, EngineThreadScope, RetiringEngine, Shutdown},
    MistralRsError, MistralRsShutdownFailure, Request,
};

use super::empty_state;

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
    let models = empty_state();
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

#[tokio::test]
async fn asynchronous_shutdown_joins_a_shared_owner_and_reuses_the_blocking_result() {
    let models = Arc::new(empty_state());
    let (first, second) = tokio::join!(models.clone().shutdown(), models.clone().shutdown());
    assert!(first.is_ok());
    assert!(second.is_ok());
    assert!(models.shutdown_blocking().is_ok());
    assert!(matches!(
        models.get_sender(None),
        Err(MistralRsError::ShuttingDown)
    ));
}

#[tokio::test]
async fn asynchronous_shutdown_retains_native_failures_for_blocking_waiters() {
    let models = Arc::new(empty_state());
    models
        .shutdown
        .record(vec![MistralRsShutdownFailure::EnginePanicked])
        .expect_err("retirement failed before explicit shutdown");
    let (first, second) = tokio::join!(models.clone().shutdown(), models.clone().shutdown());
    assert_eq!(first, second);
    let first_error = models.shutdown_blocking().expect_err("retained failure");
    let second_error = models
        .shutdown_blocking()
        .expect_err("same retained failure");
    assert_eq!(first, Err(first_error.to_string()));
    assert!(std::ptr::eq(
        first_error.failures(),
        second_error.failures()
    ));
}

#[tokio::test]
async fn cancelling_an_async_shutdown_waiter_preserves_the_in_progress_close() {
    let models = Arc::new(empty_state());
    let admission = models.shutdown.enter().expect("admitted management call");
    let waiter = tokio::spawn(models.clone().shutdown());
    let observer = models.clone();
    assert!(
        tokio::task::spawn_blocking(move || observer.shutdown.wait_for_shutdown(DEADLINE))
            .await
            .expect("shutdown observer")
    );
    waiter.abort();
    assert!(waiter.await.expect_err("cancelled waiter").is_cancelled());
    assert!(matches!(
        models.get_sender(None),
        Err(MistralRsError::ShuttingDown)
    ));
    drop(admission);
    tokio::time::timeout(DEADLINE, models.clone().shutdown())
        .await
        .expect("the original close completes")
        .expect("retained successful close");
    assert!(models.shutdown_blocking().is_ok());
}

#[test]
fn asynchronous_shutdown_rejects_native_threads_before_the_blocking_handoff() {
    let runtime = engine_runtime().expect("native runtime");
    let models = Arc::new(empty_state());
    let native_models = models.clone();
    let failure = runtime
        .block_on(runtime.spawn(async move { native_models.shutdown().await }))
        .expect("native async worker")
        .expect_err("native runtime must not wait for its own shutdown");
    assert!(failure.contains(&MistralRsShutdownFailure::EngineThreadCannotJoin.to_string()));
    assert!(models.shutdown_blocking().is_ok());
}

#[test]
fn a_closed_termination_channel_still_joins_the_native_thread() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (sender, requests) = tokio::sync::mpsc::channel(1);
    drop(requests);
    let (release, released) = mpsc::channel();
    let native = thread::spawn(move || {
        released
            .recv_timeout(DEADLINE)
            .expect("release native worker");
    });
    let engine = RetiringEngine::new(
        sender,
        Some(native),
        Resources {
            drops: drops.clone(),
            panic: false,
        },
    );
    let (healthy, healthy_terminated, healthy_release) = worker(drops.clone(), false, false);
    let shutdown = Arc::new(Shutdown::default());
    let (done, completed) = mpsc::channel();
    let waiter = {
        let shutdown = shutdown.clone();
        thread::spawn(move || {
            let result = shutdown.run(|| join_engines(vec![engine, healthy]));
            done.send(()).expect("shutdown observer");
            result
        })
    };
    healthy_terminated
        .recv_timeout(DEADLINE)
        .expect("later engine signalled after the closed channel was attempted");
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(matches!(
        completed.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    release.send(()).expect("release native worker");
    healthy_release.send(()).expect("release healthy worker");
    let failure = waiter
        .join()
        .expect("shutdown owner")
        .expect_err("closed channel");
    assert_eq!(
        failure.failures(),
        &[MistralRsShutdownFailure::TerminateChannelClosed]
    );
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}
