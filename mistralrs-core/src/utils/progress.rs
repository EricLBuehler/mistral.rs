use indicatif::{MultiProgress, ProgressBar, ProgressBarIter, ProgressIterator, ProgressStyle};
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
