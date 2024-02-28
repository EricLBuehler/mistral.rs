use crate::{response::Response, sampling::SamplingParams};
use std::sync::mpsc::Sender;

pub struct Request {
    pub prompt: String,
    pub sampling_params: SamplingParams,
    pub response: Sender<Response>,
}
