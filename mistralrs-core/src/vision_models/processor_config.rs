use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub(crate) struct ProcessorConfig {
    pub(crate) chat_template: String,
    pub(crate) image_seq_len: usize,
}
