use serde::Deserialize;

#[allow(dead_code)]
#[derive(Deserialize, Debug, Default)]
pub struct ProcessorConfig {
    pub(crate) chat_template: Option<String>,
    #[serde(alias = "image_seq_length")]
    pub(crate) image_seq_len: Option<usize>,
    pub(crate) image_break_token: Option<String>,
    pub(crate) image_end_token: Option<String>,
    pub(crate) image_token: Option<String>,
    pub(crate) patch_size: Option<usize>,
    pub(crate) spatial_merge_size: Option<usize>,
    pub(crate) pixel_shuffle_ratio: Option<f32>,
}
