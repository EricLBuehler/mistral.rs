//! Streaming event types for the OpenResponses API.

use serde::Serialize;

use super::{content::OutputContent, items::OutputItem, resource::ResponseResource};

/// All streaming event types for the OpenResponses API
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum StreamingEvent {
    /// Emitted when a response is created
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// The response resource
        response: ResponseResource,
    },
    /// Emitted when processing starts
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// The response resource
        response: ResponseResource,
    },
    /// Emitted when a new output item is added
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// The output item
        item: OutputItem,
    },
    /// Emitted when a content part is added to an output item
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// Index of the content part within the output item
        content_index: usize,
        /// The content part
        part: OutputContent,
    },
    /// Emitted when text is added to a content part
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// Index of the content part within the output item
        content_index: usize,
        /// The text delta
        delta: String,
    },
    /// Emitted when a content part is complete
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// Index of the content part within the output item
        content_index: usize,
        /// The complete content part
        part: OutputContent,
    },
    /// Emitted when an output item is complete
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// The complete output item
        item: OutputItem,
    },
    /// Emitted when function call arguments are being streamed
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// Call ID of the function call
        call_id: String,
        /// The arguments delta
        delta: String,
    },
    /// Emitted when function call arguments are complete
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Index of the output item
        output_index: usize,
        /// Call ID of the function call
        call_id: String,
        /// The complete arguments
        arguments: String,
    },
    /// Emitted when the response completes successfully
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// The final response resource
        response: ResponseResource,
    },
    /// Emitted when the response fails
    #[serde(rename = "response.failed")]
    ResponseFailed {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// The response resource with error information
        response: ResponseResource,
    },
    /// Emitted when the response is incomplete
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// The response resource with incomplete details
        response: ResponseResource,
    },
    /// Emitted when the response is cancelled
    #[serde(rename = "response.cancelled")]
    ResponseCancelled {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// The response resource
        response: ResponseResource,
    },
    /// Error event
    #[serde(rename = "error")]
    Error {
        /// Sequence number for ordering events
        sequence_number: u64,
        /// Error type
        error_type: String,
        /// Error message
        message: String,
    },
}

impl StreamingEvent {
    /// Get the sequence number of this event
    pub fn sequence_number(&self) -> u64 {
        match self {
            StreamingEvent::ResponseCreated {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ResponseInProgress {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::OutputItemAdded {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ContentPartAdded {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::OutputTextDelta {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ContentPartDone {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::OutputItemDone {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::FunctionCallArgumentsDelta {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::FunctionCallArgumentsDone {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ResponseCompleted {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ResponseFailed {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ResponseIncomplete {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::ResponseCancelled {
                sequence_number, ..
            } => *sequence_number,
            StreamingEvent::Error {
                sequence_number, ..
            } => *sequence_number,
        }
    }

    /// Get the event type string
    pub fn event_type(&self) -> &'static str {
        match self {
            StreamingEvent::ResponseCreated { .. } => "response.created",
            StreamingEvent::ResponseInProgress { .. } => "response.in_progress",
            StreamingEvent::OutputItemAdded { .. } => "response.output_item.added",
            StreamingEvent::ContentPartAdded { .. } => "response.content_part.added",
            StreamingEvent::OutputTextDelta { .. } => "response.output_text.delta",
            StreamingEvent::ContentPartDone { .. } => "response.content_part.done",
            StreamingEvent::OutputItemDone { .. } => "response.output_item.done",
            StreamingEvent::FunctionCallArgumentsDelta { .. } => {
                "response.function_call_arguments.delta"
            }
            StreamingEvent::FunctionCallArgumentsDone { .. } => {
                "response.function_call_arguments.done"
            }
            StreamingEvent::ResponseCompleted { .. } => "response.completed",
            StreamingEvent::ResponseFailed { .. } => "response.failed",
            StreamingEvent::ResponseIncomplete { .. } => "response.incomplete",
            StreamingEvent::ResponseCancelled { .. } => "response.cancelled",
            StreamingEvent::Error { .. } => "error",
        }
    }
}

/// State tracker for streaming responses
#[derive(Debug, Clone)]
pub struct StreamingState {
    /// Current sequence number
    sequence_number: u64,
    /// Response ID
    pub response_id: String,
    /// Model name
    pub model: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Current output items being built
    pub output_items: Vec<OutputItemState>,
    /// Accumulated text for each output item
    pub accumulated_text: Vec<String>,
    /// Whether the response.created event has been sent
    pub created_sent: bool,
    /// Whether the response.in_progress event has been sent
    pub in_progress_sent: bool,
}

impl StreamingState {
    /// Create a new streaming state
    pub fn new(response_id: String, model: String, created_at: u64) -> Self {
        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            output_items: Vec::new(),
            accumulated_text: Vec::new(),
            created_sent: false,
            in_progress_sent: false,
        }
    }

    /// Get the next sequence number
    pub fn next_sequence_number(&mut self) -> u64 {
        let num = self.sequence_number;
        self.sequence_number += 1;
        num
    }

    /// Get the current sequence number without incrementing
    pub fn current_sequence_number(&self) -> u64 {
        self.sequence_number
    }
}

/// State of an output item being streamed
#[derive(Debug, Clone)]
pub struct OutputItemState {
    /// Item ID
    pub id: String,
    /// Item type (message or function_call)
    pub item_type: OutputItemType,
    /// Content parts for message items
    pub content: Vec<OutputContent>,
    /// For function calls: call_id
    pub call_id: Option<String>,
    /// For function calls: function name
    pub name: Option<String>,
    /// For function calls: accumulated arguments
    pub arguments: String,
}

/// Type of output item
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputItemType {
    /// Message output
    Message,
    /// Function call output
    FunctionCall,
}
