//! ## OpenResponses API types module.
//!
//! This module contains all the types required for the OpenResponses API specification.
//! See: <https://www.openresponses.org/>

pub mod content;
pub mod enums;
pub mod events;
pub mod items;
pub mod resource;

// Re-export commonly used types
pub use content::{Annotation, InputContent, OutputContent};
pub use enums::{
    ImageDetail, IncompleteReason, ItemStatus, MessageRole, ReasoningEffort, ResponseStatus,
    ServiceTier, TruncationStrategy,
};
pub use events::{OutputItemState, OutputItemType, StreamingEvent, StreamingState};
pub use items::{InputItem, MessageContentParam, MessageItemParam, OutputItem, ResponsesInput};
pub use resource::{
    FunctionDefinition, IncompleteDetails, InputTokensDetails, OutputTokensDetails, ResponseError,
    ResponseResource, ResponseTool, ResponseToolChoice, ResponseUsage, SimpleToolChoice,
    SpecificFunctionChoice, SpecificToolChoice,
};
