use std::thread::JoinHandle;

use either::Either;
use indicatif::{ProgressBar, ProgressBarIter, ProgressIterator, ProgressStyle};
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

/// Nice progress bar with over an iterator and a message.
/// COLOR is one of r,g,b
pub struct NiceProgressBar<T: ExactSizeIterator, const COLOR: char = 'b'>(pub T, pub &'static str);

impl<T: ExactSizeIterator, const COLOR: char> IntoIterator for NiceProgressBar<T, COLOR> {
    type IntoIter = ProgressBarIter<T>;
    type Item = T::Item;

    fn into_iter(self) -> Self::IntoIter {
        let color = match COLOR {
            'b' => "blue",
            'g' => "green",
            'r' => "red",
            other => panic!("Color char `{other}` not supported"),
        };
        let bar = ProgressBar::new(self.0.len() as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{}: [{{elapsed_precise}}] [{{bar:40.{color}/{color}}}] {{pos}}/{{len}} ({{eta}})",
                    self.1
                ))
                .unwrap()
                .progress_chars("#>-"),
        );
        self.0.progress_with(bar)
    }
}
