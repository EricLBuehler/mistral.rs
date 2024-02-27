use pipeline::Pipeline;

mod models;
mod pipeline;
mod utils;

pub use models::mistralrunner::run_mistral;
pub use pipeline::{Loader, MistralLoader, MistralSpecificConfig, TokenSource};

pub struct FastServ {
    pipeline: Box<dyn Pipeline>,
}

impl FastServ {
    pub fn new() -> Self {
        todo!();
    }
}
