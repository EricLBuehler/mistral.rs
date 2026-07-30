/// Internal stand-in used only to compile non-audio paths in multimodal models.
///
/// This type is deliberately crate-private: disabling the `audio` feature removes
/// audio input from the public API instead of exposing a non-functional substitute.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct AudioInput {
    pub(crate) samples: Vec<f32>,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
}

impl AudioInput {
    pub(crate) fn to_mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let mut mono = vec![0.0; self.samples.len() / self.channels as usize];
        for (index, sample) in self.samples.iter().enumerate() {
            mono[index / self.channels as usize] += *sample;
        }
        for sample in &mut mono {
            *sample /= self.channels as f32;
        }
        mono
    }
}
