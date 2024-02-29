use std::error::Error;

use crate::sequence::StopReason;

pub enum Response {
    Error(Box<dyn Error + Send + Sync>),
    Done((StopReason, String)),
}
