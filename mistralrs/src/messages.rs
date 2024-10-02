use std::{collections::HashMap, fmt::Display, sync::Arc};

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
    fn take_sampling_params(&mut self) -> SamplingParams;
}

#[derive(Debug, Clone, PartialEq)]
/// Plain text (chat) messages.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
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
    fn take_sampling_params(&mut self) -> SamplingParams {
        SamplingParams::deterministic()
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Text (chat) messages with images.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
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

    /// This handles adding the `<|image|>` prefix to the prompt.
    pub fn add_vllama_image_message(
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
                Either::Left(format!("<|image|>{}", text.to_string())),
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
                Either::Left(format!("<image>{}", text.to_string())),
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
    fn take_sampling_params(&mut self) -> SamplingParams {
        SamplingParams::deterministic()
    }
}

#[derive(Clone)]
/// A way to add messages with finer control given.
///
/// This includes control over:
/// - Logits processors
/// - Constraints
/// - Logprobs
/// - Tools
/// - Sampling
pub struct RequestBuilder {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    adapters: Vec<String>,
    return_logprobs: bool,
    constraint: Constraint,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
    sampling_params: SamplingParams,
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
            sampling_params: SamplingParams::deterministic(),
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
            sampling_params: SamplingParams::deterministic(),
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
            sampling_params: SamplingParams::deterministic(),
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

    /// Set the sampling parameters as given.
    pub fn set_sampling(mut self, params: SamplingParams) -> Self {
        self.sampling_params = params;
        self
    }

    /// Set the sampling parameters for deterministic generation.
    /// This sets up the parameters so that there is:
    /// - No temperature, topk, topp, minp
    /// - No penalties, stop tokens, or logit bias
    /// - No maximum length
    pub fn set_deterministic_sampler(mut self) -> Self {
        self.sampling_params = SamplingParams::deterministic();
        self
    }

    pub fn set_sampler_temperature(mut self, temperature: f64) -> Self {
        self.sampling_params.temperature = Some(temperature);
        self
    }

    pub fn set_sampler_topk(mut self, topk: usize) -> Self {
        self.sampling_params.top_k = Some(topk);
        self
    }

    pub fn set_sampler_topp(mut self, topp: f64) -> Self {
        self.sampling_params.top_p = Some(topp);
        self
    }

    pub fn set_sampler_minp(mut self, minp: f64) -> Self {
        self.sampling_params.min_p = Some(minp);
        self
    }

    pub fn set_sampler_topn_logprobs(mut self, top_n_logprobs: usize) -> Self {
        self.sampling_params.top_n_logprobs = top_n_logprobs;
        self
    }

    pub fn set_sampler_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.sampling_params.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn set_sampler_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.sampling_params.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn set_sampler_stop_toks(mut self, stop_toks: StopTokens) -> Self {
        self.sampling_params.stop_toks = Some(stop_toks);
        self
    }

    pub fn set_sampler_max_len(mut self, max_len: usize) -> Self {
        self.sampling_params.max_len = Some(max_len);
        self
    }

    pub fn set_sampler_logits_bias(mut self, logits_bias: HashMap<u32, f32>) -> Self {
        self.sampling_params.logits_bias = Some(logits_bias);
        self
    }

    pub fn set_sampler_n_choices(mut self, n_choices: usize) -> Self {
        self.sampling_params.n_choices = n_choices;
        self
    }

    pub fn set_sampler_dry_params(mut self, dry_params: DrySamplingParams) -> Self {
        self.sampling_params.dry_params = Some(dry_params);
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
    fn take_sampling_params(&mut self) -> SamplingParams {
        let mut other = SamplingParams::deterministic();
        std::mem::swap(&mut other, &mut self.sampling_params);
        other
    }
}
