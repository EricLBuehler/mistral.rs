use std::{fmt::Display, sync::Arc};

use super::*;
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;

/// A type which can be used as a request.
pub trait RequestLike {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>];
    fn take_messages(&mut self) -> RequestMessage;
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>>;
    fn take_adapters(&mut self) -> Option<Vec<String>>;
    fn return_logprobs(&self) -> bool;
    fn take_constraint(&mut self) -> Constraint;
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)>;
}

#[derive(Debug, Clone, PartialEq)]
/// Plain text (chat) messages.
pub struct TextMessages(Vec<IndexMap<String, MessageContent>>);

/// A chat message role.
pub enum TextMessageRole {
    User,
    Assistant,
    System,
    Tool,
    Custom(String),
}

impl Display for TextMessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::System => write!(f, "system"),
            Self::Tool => write!(f, "tool"),
            Self::Custom(c) => write!(f, "{c}"),
        }
    }
}

impl Default for TextMessages {
    fn default() -> Self {
        Self::new()
    }
}

impl TextMessages {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.0.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    pub fn clear(mut self) -> Self {
        self.0.clear();
        self
    }
}

impl RequestLike for TextMessages {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.0
    }
    fn take_messages(&mut self) -> RequestMessage {
        let mut other = Vec::new();
        std::mem::swap(&mut other, &mut self.0);
        RequestMessage::Chat(other)
    }
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>> {
        None
    }
    fn take_adapters(&mut self) -> Option<Vec<String>> {
        None
    }
    fn return_logprobs(&self) -> bool {
        false
    }
    fn take_constraint(&mut self) -> Constraint {
        Constraint::None
    }
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)> {
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Text (chat) messages with images.
pub struct VisionMessages {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
}

impl Default for VisionMessages {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionMessages {
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            messages: Vec::new(),
        }
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// This handles adding the `<|image_{N}|>` prefix to the prompt.
    pub fn add_phiv_image_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        image: DynamicImage,
    ) -> Self {
        self.images.push(image);
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            (
                "content".to_string(),
                Either::Left(format!(
                    "<|image_{}|>{}",
                    self.images.len(),
                    text.to_string()
                )),
            ),
        ]));
        self
    }

    /// This handles adding the `<image>` prefix to the prompt.
    pub fn add_llava_image_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        image: DynamicImage,
    ) -> Self {
        self.images.push(image);
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            (
                "content".to_string(),
                Either::Left(format!(
                    "<|image_{}|>{}",
                    self.images.len(),
                    text.to_string()
                )),
            ),
        ]));
        self
    }

    pub fn add_idefics_image_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        image: DynamicImage,
    ) -> Self {
        self.images.push(image);
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            (
                "content".to_string(),
                Either::Right(vec![
                    IndexMap::from([("type".to_string(), "image".to_string())]),
                    IndexMap::from([
                        ("type".to_string(), "text".to_string()),
                        ("content".to_string(), text.to_string()),
                    ]),
                ]),
            ),
        ]));
        self
    }

    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self.images.clear();

        self
    }
}

impl RequestLike for VisionMessages {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }
    fn take_messages(&mut self) -> RequestMessage {
        let mut other_messages = Vec::new();
        std::mem::swap(&mut other_messages, &mut self.messages);
        let mut other_images = Vec::new();
        std::mem::swap(&mut other_images, &mut self.images);
        RequestMessage::VisionChat {
            images: other_images,
            messages: other_messages,
        }
    }
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>> {
        None
    }
    fn take_adapters(&mut self) -> Option<Vec<String>> {
        None
    }
    fn return_logprobs(&self) -> bool {
        false
    }
    fn take_constraint(&mut self) -> Constraint {
        Constraint::None
    }
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)> {
        None
    }
}

#[derive(Clone)]
/// A way to add messages with finer control given.
pub struct RequestBuilder {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    adapters: Vec<String>,
    return_logprobs: bool,
    constraint: Constraint,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl From<TextMessages> for RequestBuilder {
    fn from(value: TextMessages) -> Self {
        Self {
            messages: value.0,
            images: Vec::new(),
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
        }
    }
}

impl From<VisionMessages> for RequestBuilder {
    fn from(value: VisionMessages) -> Self {
        Self {
            messages: value.messages,
            images: value.images,
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
        }
    }
}

impl RequestBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            images: Vec::new(),
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
        }
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    pub fn add_image_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        image: DynamicImage,
    ) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self.images.push(image);
        self
    }

    pub fn add_logits_processor(mut self, processor: Arc<dyn CustomLogitsProcessor>) -> Self {
        self.logits_processors.push(processor);
        self
    }

    pub fn set_adapters(mut self, adapters: Vec<String>) -> Self {
        self.adapters = adapters;
        self
    }

    /// The default tool choice is auto.
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn set_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    pub fn return_logprobs(mut self, return_logprobs: bool) -> Self {
        self.return_logprobs = return_logprobs;
        self
    }

    pub fn set_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = constraint;
        self
    }
}

impl RequestLike for RequestBuilder {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }

    fn take_messages(&mut self) -> RequestMessage {
        if self.images.is_empty() {
            let mut other = Vec::new();
            std::mem::swap(&mut other, &mut self.messages);
            RequestMessage::Chat(other)
        } else {
            let mut other_messages = Vec::new();
            std::mem::swap(&mut other_messages, &mut self.messages);
            let mut other_images = Vec::new();
            std::mem::swap(&mut other_images, &mut self.images);
            RequestMessage::VisionChat {
                images: other_images,
                messages: other_messages,
            }
        }
    }

    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>> {
        if self.logits_processors.is_empty() {
            None
        } else {
            let mut other = Vec::new();
            std::mem::swap(&mut other, &mut self.logits_processors);
            Some(other)
        }
    }

    fn take_adapters(&mut self) -> Option<Vec<String>> {
        if self.adapters.is_empty() {
            None
        } else {
            let mut other = Vec::new();
            std::mem::swap(&mut other, &mut self.adapters);
            Some(other)
        }
    }

    fn return_logprobs(&self) -> bool {
        self.return_logprobs
    }

    fn take_constraint(&mut self) -> Constraint {
        let mut other = Constraint::None;
        std::mem::swap(&mut other, &mut self.constraint);
        other
    }

    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)> {
        if self.tools.is_empty() {
            None
        } else {
            let mut other_ts = Vec::new();
            std::mem::swap(&mut other_ts, &mut self.tools);
            let mut other_tc = ToolChoice::Auto;
            std::mem::swap(&mut other_tc, &mut self.tool_choice);
            Some((other_ts, other_tc))
        }
    }
}
