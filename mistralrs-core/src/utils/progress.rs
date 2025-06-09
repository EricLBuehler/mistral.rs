use indicatif::{MultiProgress, ProgressBar, ProgressBarIter, ProgressIterator, ProgressStyle};
use mistralrs_quant::get_immediate_isq;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::iter::Iterator;
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

/// Nice progress bar with over an iterator and a message.
/// COLOR is one of r,g,b
pub struct NiceProgressBar<'a, T: ExactSizeIterator, const COLOR: char = 'b'>(
    pub T,
    pub &'static str,
    pub &'a MultiProgress,
);

impl<T: ExactSizeIterator, const COLOR: char> IntoIterator for NiceProgressBar<'_, T, COLOR> {
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

        // Add to the multi progress
        self.2.add(bar.clone());

        self.0.progress_with(bar)
    }
}

/// Parallel iterator with progress reporting.
pub struct ParProgress<I> {
    iter: I,
    bar: ProgressBar,
}

impl<I> ParallelIterator for ParProgress<I>
where
    I: ParallelIterator,
    I::Item: Send,
{
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            item
        });
        iter.drive_unindexed(consumer)
    }
}

impl<I> IndexedParallelIterator for ParProgress<I>
where
    I: IndexedParallelIterator,
    I::Item: Send,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            item
        });
        iter.drive(consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        let bar = self.bar.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            item
        });
        iter.with_producer(callback)
    }
}

impl<'a, T, const COLOR: char> IntoParallelIterator for NiceProgressBar<'a, T, COLOR>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: ParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + Send,
{
    type Iter = ParProgress<T::Iter>;
    type Item = <T as IntoParallelIterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
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
        self.2.add(bar.clone());
        ParProgress {
            iter: self.0.into_par_iter(),
            bar,
        }
    }
}

impl<'a, T, const COLOR: char> NiceProgressBar<'a, T, COLOR>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: ParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + Send,
    T: IntoParallelIterator<Item = <T as Iterator>::Item>,
{
    /// Applies the given closure over the items, optionally in parallel, and collects the results.
    ///
    /// - `is_parallel`: If true, uses Rayon parallel iteration; otherwise uses sequential iteration.
    /// - `f`: A closure to apply to each item.
    pub fn run<F, U>(self, _is_parallel: bool, f: F) -> candle_core::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> candle_core::Result<U> + Sync + Send,
        U: Send,
    {
        // if is_parallel {
        //     self.into_par_iter().map(f).collect()
        // } else {
        //     self.into_iter().map(f).collect()
        // }
        self.into_iter().map(f).collect()
    }

    /// Applies the given closure over the items, optionally in parallel, and collects the results.
    ///
    /// - `f`: A closure to apply to each item.
    pub fn par_iter_if_isq<F, U>(self, f: F) -> candle_core::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> candle_core::Result<U> + Sync + Send,
        U: Send,
    {
        self.run(get_immediate_isq().is_some_and(|x| x.ty.is_some()), f)
    }
}
