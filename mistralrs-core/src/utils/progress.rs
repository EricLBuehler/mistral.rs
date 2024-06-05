use std::thread::JoinHandle;

use either::Either;
use tqdm::Iter;

// Optionally display a progress bar via the `tqdm` crate:
// Usage: `iter.with_progress(true)`
// Similar to the `iter.tqdm()` feature except this supports opt-in via parameter.
pub trait IterWithProgress<'a, T>: Iterator<Item = T> + 'a {
    fn with_progress(self, is_silent: bool) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: Sized,
    {
        // TODO: Should `is_silent` instead be referenced as a global read-only state? (`AtomicBool`)
        if is_silent {
            Box::new(self)
        } else {
            Box::new(self.tqdm())
        }
    }
}

impl<'a, T: Iterator + 'a> IterWithProgress<'a, T::Item> for T {}

/// Choose between threading or non-threading depending on if the `metal`
/// feature is enabled.
pub struct Parellelize;

/// A handle which does not do threading. Instead, it always reports that is is
/// finished and executes the closure lazily. This is used for Metal
/// where the command buffer cannot be used concurrently.
pub struct NonThreadingHandle<T, F>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    f: F,
}

impl<T, F> NonThreadingHandle<T, F>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    fn join(self) -> std::thread::Result<T> {
        std::thread::Result::Ok((self.f)())
    }
    fn is_finished(&self) -> bool {
        true
    }
}

/// A trait representing a joinable handle.
pub trait Joinable<T> {
    fn join(self) -> std::thread::Result<T>;
    fn is_finished(&self) -> bool;
}

impl<T, F> Joinable<T> for Either<JoinHandle<T>, NonThreadingHandle<T, F>>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    fn is_finished(&self) -> bool {
        match self {
            Self::Left(l) => l.is_finished(),
            Self::Right(r) => r.is_finished(),
        }
    }
    fn join(self) -> std::thread::Result<T> {
        match self {
            Self::Left(l) => l.join(),
            Self::Right(r) => r.join(),
        }
    }
}

#[cfg(not(feature = "metal"))]
impl Parellelize {
    pub fn spawn<F, T>(f: F) -> Either<JoinHandle<T>, NonThreadingHandle<T, F>>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        Either::Left(std::thread::spawn(f))
    }
}

#[cfg(feature = "metal")]
impl Parellelize {
    pub fn spawn<F, T>(f: F) -> Either<JoinHandle<T>, NonThreadingHandle<T, F>>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        Either::Right(NonThreadingHandle { f })
    }
}
