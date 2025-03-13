use serde::Deserialize;

#[allow(dead_code)]
#[derive(Deserialize, Debug, Default)]
pub struct ProcessorConfig {
    pub(crate) chat_template: Option<String>,
    #[serde(alias = "image_seq_length")]
    pub(crate) image_seq_len: Option<usize>,
}
