use serde::Deserialize;

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct ProcessorConfig {
    pub(crate) chat_template: Option<String>,
    pub(crate) image_seq_len: usize,
}
