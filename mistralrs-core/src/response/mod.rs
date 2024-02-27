use std::error::Error;

pub enum Response {
    Error(Box<dyn Error + Send + Sync>),
}
