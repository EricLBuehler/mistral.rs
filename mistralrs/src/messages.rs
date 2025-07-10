use std::{collections::HashMap, fmt::Display, sync::Arc};

use super::*;
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use serde_json::{json, Value};

/// A type which can be used as a chat request.
pub trait RequestLike {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>];
    fn images_ref(&self) -> &[DynamicImage];
    fn take_messages(&mut self) -> RequestMessage;
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>>;
    fn take_adapters(&mut self) -> Option<Vec<String>>;
    fn return_logprobs(&self) -> bool;
    fn enable_search(&self) -> Option<bool>;
    fn take_constraint(&mut self) -> Constraint;
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)>;
    fn take_sampling_params(&mut self) -> SamplingParams;
    fn take_web_search_options(&mut self) -> Option<WebSearchOptions>;
}

#[derive(Debug, Clone, PartialEq)]
/// Plain text (chat) messages.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
pub struct TextMessages {
    messages: Vec<IndexMap<String, MessageContent>>,
    enable_thinking: Option<bool>,
}

impl From<TextMessages> for Vec<IndexMap<String, MessageContent>> {
    fn from(value: TextMessages) -> Self {
        value.messages
    }
}

#[derive(Debug, Clone, PartialEq)]
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
        Self {
            messages: Vec::new(),
            enable_thinking: None,
        }
    }

    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self
    }

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }
}

impl RequestLike for TextMessages {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }
    fn images_ref(&self) -> &[DynamicImage] {
        &[]
    }
    fn take_messages(&mut self) -> RequestMessage {
        let mut other = Vec::new();
        std::mem::swap(&mut other, &mut self.messages);
        RequestMessage::Chat {
            messages: other,
            enable_thinking: self.enable_thinking,
        }
    }
    fn enable_search(&self) -> Option<bool> {
        None
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
    fn take_web_search_options(&mut self) -> Option<WebSearchOptions> {
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Text (chat) messages with images and/or audios.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
pub struct VisionMessages {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    audios: Vec<AudioInput>,
    enable_thinking: Option<bool>,
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
            audios: Vec::new(),
            enable_thinking: None,
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
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, images, vec![], model)
    }

    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, vec![], audios, model)
    }

    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        let config = model.config().unwrap();
        let prefixer = match &config.category {
            ModelCategory::Vision { prefixer } => prefixer,
            ModelCategory::Text
            | ModelCategory::Diffusion
            | ModelCategory::Speech
            | ModelCategory::Audio => {
                anyhow::bail!("`add_image_message` expects a vision model.")
            }
        };

        // Images
        let n_added_images = images.len();
        let prefixed = prefixer.prefix_image(
            (self.images.len()..self.images.len() + n_added_images).collect(),
            &text.to_string(),
        );
        self.images.extend(images);

        // Audios
        let n_added_audios = audios.len();
        let prefixed = prefixer.prefix_audio(
            (self.audios.len()..self.audios.len() + n_added_audios).collect(),
            &prefixed,
        );
        self.audios.extend(audios);

        if n_added_images > 0 {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                (
                    "content".to_string(),
                    Either::Right(vec![
                        IndexMap::from([("type".to_string(), Value::String("image".to_string()))]),
                        IndexMap::from([
                            ("type".to_string(), Value::String("text".to_string())),
                            ("text".to_string(), Value::String(prefixed)),
                        ]),
                    ]),
                ),
            ]));
        } else {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(prefixed)),
            ]));
        }
        Ok(self)
    }

    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self.images.clear();

        self
    }

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }
}

impl RequestLike for VisionMessages {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }
    fn images_ref(&self) -> &[DynamicImage] {
        &self.images
    }
    fn take_messages(&mut self) -> RequestMessage {
        let mut other_messages = Vec::new();
        std::mem::swap(&mut other_messages, &mut self.messages);
        let mut other_images = Vec::new();
        std::mem::swap(&mut other_images, &mut self.images);
        let mut other_audios = Vec::new();
        std::mem::swap(&mut other_audios, &mut self.audios);
        RequestMessage::VisionChat {
            images: other_images,
            messages: other_messages,
            audios: other_audios,
            enable_thinking: self.enable_thinking,
        }
    }
    fn enable_search(&self) -> Option<bool> {
        None
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
    fn take_web_search_options(&mut self) -> Option<WebSearchOptions> {
        None
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
/// - Enable thinking for models that support the configuration
pub struct RequestBuilder {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    audios: Vec<AudioInput>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    adapters: Vec<String>,
    return_logprobs: bool,
    constraint: Constraint,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
    sampling_params: SamplingParams,
    web_search_options: Option<WebSearchOptions>,
    enable_thinking: Option<bool>,
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl From<TextMessages> for RequestBuilder {
    fn from(value: TextMessages) -> Self {
        Self {
            messages: value.messages,
            images: Vec::new(),
            audios: Vec::new(),
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: SamplingParams::deterministic(),
            web_search_options: None,
            enable_thinking: None,
        }
    }
}

impl From<VisionMessages> for RequestBuilder {
    fn from(value: VisionMessages) -> Self {
        Self {
            messages: value.messages,
            images: value.images,
            audios: value.audios,
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: SamplingParams::deterministic(),
            web_search_options: None,
            enable_thinking: None,
        }
    }
}

impl RequestBuilder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            images: Vec::new(),
            audios: Vec::new(),
            logits_processors: Vec::new(),
            adapters: Vec::new(),
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: SamplingParams::deterministic(),
            web_search_options: None,
            enable_thinking: None,
        }
    }

    pub fn with_web_search_options(mut self, web_search_options: WebSearchOptions) -> Self {
        self.web_search_options = Some(web_search_options);
        self
    }

    /// Add a message to the request.
    ///
    /// For messages with tool calls, use [`Self::add_message_with_tool_call`].
    /// For messages with tool outputs, use [`Self::add_tool_message`].
    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// Add a message with the output of a tool call.
    pub fn add_tool_message(mut self, tool_content: impl ToString, tool_id: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            (
                "role".to_string(),
                Either::Left(TextMessageRole::Tool.to_string()),
            ),
            (
                "content".to_string(),
                Either::Left(tool_content.to_string()),
            ),
            (
                "tool_call_id".to_string(),
                Either::Left(tool_id.to_string()),
            ),
        ]));
        self
    }

    pub fn add_message_with_tool_call(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        tool_calls: Vec<ToolCallResponse>,
    ) -> Self {
        let tool_messages = tool_calls
            .iter()
            .map(|t| {
                IndexMap::from([
                    ("id".to_string(), Value::String(t.id.clone())),
                    ("type".to_string(), Value::String(t.tp.to_string())),
                    (
                        "function".to_string(),
                        json!({
                            "name": t.function.name,
                            "arguments": t.function.arguments,
                        }),
                    ),
                ])
            })
            .collect();
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
            ("function".to_string(), Either::Right(tool_messages)),
        ]));
        self
    }

    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, images, vec![], model)
    }

    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        self.add_multimodal_message(role, text, vec![], audios, model)
    }

    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        audios: Vec<AudioInput>,
        model: &Model,
    ) -> anyhow::Result<Self> {
        let config = model.config().unwrap();
        let prefixer = match &config.category {
            ModelCategory::Vision { prefixer } => prefixer,
            ModelCategory::Text
            | ModelCategory::Diffusion
            | ModelCategory::Speech
            | ModelCategory::Audio => {
                anyhow::bail!("`add_image_message` expects a vision model.")
            }
        };

        // Images
        let n_added_images = images.len();
        let prefixed = prefixer.prefix_image(
            (self.images.len()..self.images.len() + n_added_images).collect(),
            &text.to_string(),
        );
        self.images.extend(images);

        // Audios
        let n_added_audios = audios.len();
        let prefixed = prefixer.prefix_audio(
            (self.audios.len()..self.audios.len() + n_added_audios).collect(),
            &prefixed,
        );
        self.audios.extend(audios);

        if n_added_images > 0 {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                (
                    "content".to_string(),
                    Either::Right(vec![
                        IndexMap::from([("type".to_string(), Value::String("image".to_string()))]),
                        IndexMap::from([
                            ("type".to_string(), Value::String("text".to_string())),
                            ("text".to_string(), Value::String(prefixed)),
                        ]),
                    ]),
                ),
            ]));
        } else {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(prefixed)),
            ]));
        }
        Ok(self)
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

    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }
}

impl RequestLike for RequestBuilder {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }

    fn images_ref(&self) -> &[DynamicImage] {
        &self.images
    }

    fn take_messages(&mut self) -> RequestMessage {
        if self.images.is_empty() && self.audios.is_empty() {
            let mut other = Vec::new();
            std::mem::swap(&mut other, &mut self.messages);
            RequestMessage::Chat {
                messages: other,
                enable_thinking: self.enable_thinking,
            }
        } else {
            let mut other_messages = Vec::new();
            std::mem::swap(&mut other_messages, &mut self.messages);
            let mut other_images = Vec::new();
            std::mem::swap(&mut other_images, &mut self.images);
            let mut other_audios = Vec::new();
            std::mem::swap(&mut other_audios, &mut self.audios);
            RequestMessage::VisionChat {
                images: other_images,
                messages: other_messages,
                audios: other_audios,
                enable_thinking: self.enable_thinking,
            }
        }
    }

    fn enable_search(&self) -> Option<bool> {
        self.enable_thinking
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

    fn take_web_search_options(&mut self) -> Option<WebSearchOptions> {
        let mut other = None;
        std::mem::swap(&mut other, &mut self.web_search_options);
        other
    }
}
