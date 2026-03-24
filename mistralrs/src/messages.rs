use std::{collections::HashMap, fmt::Display, sync::Arc};

use super::*;
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use serde_json::{json, Value};

/// A type which can be used as a chat request.
///
/// Implemented by [`TextMessages`], [`VisionMessages`], and [`RequestBuilder`].
pub trait RequestLike {
    /// Borrow the current list of chat messages.
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>];
    /// Borrow the current list of images (empty for text-only requests).
    fn images_ref(&self) -> &[DynamicImage];
    /// Take ownership of the messages, converting them into a [`RequestMessage`].
    fn take_messages(&mut self) -> RequestMessage;
    /// Take any custom logits processors, if configured.
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>>;
    /// Take any active adapter names (LoRA / X-LoRA), if configured.
    fn take_adapters(&mut self) -> Option<Vec<String>>;
    /// Whether log-probabilities should be returned.
    fn return_logprobs(&self) -> bool;
    /// Whether web search should be enabled for this request.
    fn enable_search(&self) -> Option<bool>;
    /// Take the generation constraint (regex, JSON schema, grammar, or none).
    fn take_constraint(&mut self) -> Constraint;
    /// Take the tools and tool-choice policy, if any.
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)>;
    /// Take the sampling parameters.
    fn take_sampling_params(&mut self) -> SamplingParams;
    /// Take web search options, if configured.
    fn take_web_search_options(&mut self) -> Option<WebSearchOptions>;
    /// Whether to silently truncate prompts that exceed the model's context length.
    fn truncate_sequence(&self) -> bool {
        false
    }
    /// Apply any deferred model-specific media prefixes.
    ///
    /// Called automatically by [`Model`](crate::Model) before sending the request.
    /// The default implementation is a no-op.
    fn resolve_pending_prefixes(&mut self, _category: &ModelCategory) {}
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
    /// The human user.
    User,
    /// The model / assistant.
    Assistant,
    /// System prompt providing instructions to the model.
    System,
    /// Output from a tool call.
    Tool,
    /// A custom role string.
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
    /// Create an empty text message list.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            enable_thinking: None,
        }
    }

    /// Append a message with the given role and text content.
    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// Remove all messages.
    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self
    }

    /// Enable extended thinking (chain-of-thought) for models that support it.
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
            reasoning_effort: None,
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

impl From<TextMessages> for VisionMessages {
    fn from(text: TextMessages) -> Self {
        Self {
            messages: text.messages,
            images: Vec::new(),
            audios: Vec::new(),
            enable_thinking: text.enable_thinking,
            pending_prefixes: Vec::new(),
        }
    }
}

/// Tracks a message whose text has not yet been prefixed with model-specific
/// media tokens. Resolved automatically at send-time.
#[derive(Debug, Clone, PartialEq)]
struct PendingMediaPrefix {
    /// Index into the owning struct's `messages` vec.
    message_index: usize,
    /// Global image indices that belong to this message.
    image_indices: Vec<usize>,
    /// Global audio indices that belong to this message.
    audio_indices: Vec<usize>,
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
    pending_prefixes: Vec<PendingMediaPrefix>,
}

impl Default for VisionMessages {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionMessages {
    /// Create an empty vision message list.
    pub fn new() -> Self {
        Self {
            images: Vec::new(),
            messages: Vec::new(),
            audios: Vec::new(),
            enable_thinking: None,
            pending_prefixes: Vec::new(),
        }
    }

    /// Append a text-only message with the given role and content.
    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// Append a message containing images.
    ///
    /// Model-specific prefix tokens are applied automatically when the
    /// request is sent via [`Model::send_chat_request`](crate::Model::send_chat_request)
    /// or [`Model::stream_chat_request`](crate::Model::stream_chat_request).
    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
    ) -> Self {
        self.add_multimodal_message(role, text, images, vec![])
    }

    /// Append a message containing audio.
    ///
    /// Model-specific prefix tokens are applied automatically when the
    /// request is sent via [`Model::send_chat_request`](crate::Model::send_chat_request)
    /// or [`Model::stream_chat_request`](crate::Model::stream_chat_request).
    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
    ) -> Self {
        self.add_multimodal_message(role, text, vec![], audios)
    }

    /// Append a message containing a mix of text, images, and/or audio.
    ///
    /// Model-specific prefix tokens are applied automatically when the
    /// request is sent via [`Model::send_chat_request`](crate::Model::send_chat_request)
    /// or [`Model::stream_chat_request`](crate::Model::stream_chat_request).
    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        audios: Vec<AudioInput>,
    ) -> Self {
        // Images
        let n_added_images = images.len();
        let image_indices: Vec<usize> =
            (self.images.len()..self.images.len() + n_added_images).collect();
        self.images.extend(images);

        // Audios
        let n_added_audios = audios.len();
        let audio_indices: Vec<usize> =
            (self.audios.len()..self.audios.len() + n_added_audios).collect();
        self.audios.extend(audios);

        if n_added_images > 0 || n_added_audios > 0 {
            let mut content_vec: Vec<IndexMap<String, Value>> = Vec::new();
            for _ in 0..n_added_images {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("image".to_string()),
                )]));
            }
            for _ in 0..n_added_audios {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("audio".to_string()),
                )]));
            }
            // Store raw (unprefixed) text — prefixing happens at send-time
            content_vec.push(IndexMap::from([
                ("type".to_string(), Value::String("text".to_string())),
                ("text".to_string(), Value::String(text.to_string())),
            ]));

            let message_index = self.messages.len();
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Right(content_vec)),
            ]));

            self.pending_prefixes.push(PendingMediaPrefix {
                message_index,
                image_indices,
                audio_indices,
            });
        } else {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(text.to_string())),
            ]));
        }
        self
    }

    /// Remove all messages, images, and audio.
    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self.images.clear();
        self.audios.clear();
        self.pending_prefixes.clear();
        self
    }

    /// Enable extended thinking (chain-of-thought) for models that support it.
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
    fn resolve_pending_prefixes(&mut self, category: &ModelCategory) {
        resolve_pending(category, &mut self.messages, &mut self.pending_prefixes);
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
            reasoning_effort: None,
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
    truncate_sequence: bool,
    pending_prefixes: Vec<PendingMediaPrefix>,
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
            truncate_sequence: false,
            pending_prefixes: Vec::new(),
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
            truncate_sequence: false,
            pending_prefixes: value.pending_prefixes,
        }
    }
}

impl RequestBuilder {
    /// Create an empty request builder with deterministic sampling defaults.
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
            truncate_sequence: false,
            pending_prefixes: Vec::new(),
        }
    }

    /// Enable web search with the given options.
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

    /// Append an assistant message that includes tool call results.
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

    /// Append a message containing images.
    ///
    /// Model-specific prefix tokens are applied automatically when the
    /// request is sent via [`Model::send_chat_request`](crate::Model::send_chat_request)
    /// or [`Model::stream_chat_request`](crate::Model::stream_chat_request).
    pub fn add_image_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
    ) -> Self {
        self.add_multimodal_message(role, text, images, vec![])
    }

    /// Append a message containing audio.
    ///
    /// Model-specific prefix tokens are applied automatically when the
    /// request is sent via [`Model::send_chat_request`](crate::Model::send_chat_request)
    /// or [`Model::stream_chat_request`](crate::Model::stream_chat_request).
    pub fn add_audio_message(
        self,
        role: TextMessageRole,
        text: impl ToString,
        audios: Vec<AudioInput>,
    ) -> Self {
        self.add_multimodal_message(role, text, vec![], audios)
    }

    /// Append a message containing a mix of text, images, and/or audio.
    ///
    /// Model-specific prefix tokens are applied automatically when the
    /// request is sent via [`Model::send_chat_request`](crate::Model::send_chat_request)
    /// or [`Model::stream_chat_request`](crate::Model::stream_chat_request).
    pub fn add_multimodal_message(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        images: Vec<DynamicImage>,
        audios: Vec<AudioInput>,
    ) -> Self {
        // Images
        let n_added_images = images.len();
        let image_indices: Vec<usize> =
            (self.images.len()..self.images.len() + n_added_images).collect();
        self.images.extend(images);

        // Audios
        let n_added_audios = audios.len();
        let audio_indices: Vec<usize> =
            (self.audios.len()..self.audios.len() + n_added_audios).collect();
        self.audios.extend(audios);

        if n_added_images > 0 || n_added_audios > 0 {
            let mut content_vec: Vec<IndexMap<String, Value>> = Vec::new();
            for _ in 0..n_added_images {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("image".to_string()),
                )]));
            }
            for _ in 0..n_added_audios {
                content_vec.push(IndexMap::from([(
                    "type".to_string(),
                    Value::String("audio".to_string()),
                )]));
            }
            // Store raw (unprefixed) text — prefixing happens at send-time
            content_vec.push(IndexMap::from([
                ("type".to_string(), Value::String("text".to_string())),
                ("text".to_string(), Value::String(text.to_string())),
            ]));

            let message_index = self.messages.len();
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Right(content_vec)),
            ]));

            self.pending_prefixes.push(PendingMediaPrefix {
                message_index,
                image_indices,
                audio_indices,
            });
        } else {
            self.messages.push(IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(text.to_string())),
            ]));
        }
        self
    }

    /// Add a custom logits processor applied during token sampling.
    pub fn add_logits_processor(mut self, processor: Arc<dyn CustomLogitsProcessor>) -> Self {
        self.logits_processors.push(processor);
        self
    }

    /// Activate the given LoRA/X-LoRA adapter layers by name.
    pub fn set_adapters(mut self, adapters: Vec<String>) -> Self {
        self.adapters = adapters;
        self
    }

    /// The default tool choice is auto.
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// Control how the model selects tools (auto, required, none, or a specific tool).
    pub fn set_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Request log-probabilities for each generated token.
    pub fn return_logprobs(mut self, return_logprobs: bool) -> Self {
        self.return_logprobs = return_logprobs;
        self
    }

    /// Apply a generation constraint (regex, JSON schema, or grammar).
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

    /// Set the sampling temperature. Higher values increase randomness.
    pub fn set_sampler_temperature(mut self, temperature: f64) -> Self {
        self.sampling_params.temperature = Some(temperature);
        self
    }

    /// Limit sampling to the top-k most probable tokens.
    pub fn set_sampler_topk(mut self, topk: usize) -> Self {
        self.sampling_params.top_k = Some(topk);
        self
    }

    /// Nucleus sampling: only consider tokens whose cumulative probability exceeds this threshold.
    pub fn set_sampler_topp(mut self, topp: f64) -> Self {
        self.sampling_params.top_p = Some(topp);
        self
    }

    /// Min-p sampling: filter tokens below this fraction of the top token's probability.
    pub fn set_sampler_minp(mut self, minp: f64) -> Self {
        self.sampling_params.min_p = Some(minp);
        self
    }

    /// Return the top-n log-probabilities per token position.
    pub fn set_sampler_topn_logprobs(mut self, top_n_logprobs: usize) -> Self {
        self.sampling_params.top_n_logprobs = top_n_logprobs;
        self
    }

    /// Penalize tokens proportionally to how often they have appeared so far.
    pub fn set_sampler_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.sampling_params.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Penalize tokens that have appeared at all, regardless of frequency.
    pub fn set_sampler_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.sampling_params.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set stop tokens that terminate generation when produced.
    pub fn set_sampler_stop_toks(mut self, stop_toks: StopTokens) -> Self {
        self.sampling_params.stop_toks = Some(stop_toks);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn set_sampler_max_len(mut self, max_len: usize) -> Self {
        self.sampling_params.max_len = Some(max_len);
        self
    }

    /// Apply a bias to specific token IDs during sampling.
    pub fn set_sampler_logits_bias(mut self, logits_bias: HashMap<u32, f32>) -> Self {
        self.sampling_params.logits_bias = Some(logits_bias);
        self
    }

    /// Generate multiple independent completions for the same prompt.
    pub fn set_sampler_n_choices(mut self, n_choices: usize) -> Self {
        self.sampling_params.n_choices = n_choices;
        self
    }

    /// Configure DRY (Don't Repeat Yourself) sampling parameters.
    pub fn set_sampler_dry_params(mut self, dry_params: DrySamplingParams) -> Self {
        self.sampling_params.dry_params = Some(dry_params);
        self
    }

    /// Enable extended thinking (chain-of-thought) for models that support it.
    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Truncate prompts that exceed the model's maximum context length.
    pub fn with_truncate_sequence(mut self, truncate_sequence: bool) -> Self {
        self.truncate_sequence = truncate_sequence;
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

    fn resolve_pending_prefixes(&mut self, category: &ModelCategory) {
        resolve_pending(category, &mut self.messages, &mut self.pending_prefixes);
    }

    fn take_messages(&mut self) -> RequestMessage {
        if self.images.is_empty() && self.audios.is_empty() {
            let mut other = Vec::new();
            std::mem::swap(&mut other, &mut self.messages);
            RequestMessage::Chat {
                messages: other,
                enable_thinking: self.enable_thinking,
                reasoning_effort: None,
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
                reasoning_effort: None,
            }
        }
    }

    fn enable_search(&self) -> Option<bool> {
        self.web_search_options.as_ref().map(|_| true)
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

    fn truncate_sequence(&self) -> bool {
        self.truncate_sequence
    }
}

/// Shared implementation: apply a model-specific prefixer to all pending messages.
fn resolve_pending(
    category: &ModelCategory,
    messages: &mut [IndexMap<String, MessageContent>],
    pending: &mut Vec<PendingMediaPrefix>,
) {
    let prefixer = match category {
        ModelCategory::Vision { prefixer } => prefixer,
        _ => {
            // Not a vision model — nothing to prefix.
            pending.clear();
            return;
        }
    };

    for entry in pending.drain(..) {
        let Some(msg) = messages.get_mut(entry.message_index) else {
            continue;
        };
        let Some(Either::Right(content_vec)) = msg.get_mut("content") else {
            continue;
        };
        // Find the text part and apply prefixing
        for part in content_vec.iter_mut() {
            let is_text = part
                .get("type")
                .is_some_and(|v| v == &Value::String("text".to_string()));
            if !is_text {
                continue;
            }
            if let Some(Value::String(text)) = part.get_mut("text") {
                if !entry.image_indices.is_empty() {
                    *text = prefixer.prefix_image(entry.image_indices.clone(), text);
                }
                if !entry.audio_indices.is_empty() {
                    *text = prefixer.prefix_audio(entry.audio_indices.clone(), text);
                }
            }
            break;
        }
    }
}

#[derive(Clone, Debug)]
/// An individual embedding input.
pub enum EmbeddingRequestInput {
    /// Raw text prompt that will be tokenized.
    Prompt(String),
    /// Pre-tokenized input.
    Tokens(Vec<u32>),
}

impl EmbeddingRequestInput {
    /// Convert this input into a [`RequestMessage`] suitable for the engine.
    pub fn into_request_message(self) -> RequestMessage {
        match self {
            Self::Prompt(prompt) => RequestMessage::Embedding { prompt },
            Self::Tokens(prompt) => RequestMessage::EmbeddingTokens { prompt },
        }
    }
}

#[derive(Clone, Debug)]
/// A validated embedding request constructed via [`EmbeddingRequestBuilder`].
pub struct EmbeddingRequest {
    /// The embedding inputs (text prompts or pre-tokenized sequences).
    pub inputs: Vec<EmbeddingRequestInput>,
    /// Whether to truncate inputs that exceed the model's maximum context length.
    pub truncate_sequence: bool,
}

impl EmbeddingRequest {
    /// Create a new builder for an embedding request.
    pub fn builder() -> EmbeddingRequestBuilder {
        EmbeddingRequestBuilder::new()
    }
}

/// Builder for configuring embedding requests.
#[derive(Clone, Debug, Default)]
pub struct EmbeddingRequestBuilder {
    inputs: Vec<EmbeddingRequestInput>,
    truncate_sequence: bool,
}

impl EmbeddingRequestBuilder {
    /// Create an empty builder. You must add at least one input before using it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single text prompt.
    pub fn add_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inputs
            .push(EmbeddingRequestInput::Prompt(prompt.into()));
        self
    }

    /// Add multiple text prompts at once.
    pub fn add_prompts<I, S>(mut self, prompts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inputs.extend(
            prompts
                .into_iter()
                .map(|prompt| EmbeddingRequestInput::Prompt(prompt.into())),
        );
        self
    }

    /// Add a single pre-tokenized prompt.
    pub fn add_tokens(mut self, tokens: impl Into<Vec<u32>>) -> Self {
        self.inputs
            .push(EmbeddingRequestInput::Tokens(tokens.into()));
        self
    }

    /// Add multiple pre-tokenized prompts.
    pub fn add_tokens_batch<I>(mut self, batches: I) -> Self
    where
        I: IntoIterator<Item = Vec<u32>>,
    {
        self.inputs
            .extend(batches.into_iter().map(EmbeddingRequestInput::Tokens));
        self
    }

    /// Control whether prompts longer than the model context are truncated.
    pub fn with_truncate_sequence(mut self, truncate: bool) -> Self {
        self.truncate_sequence = truncate;
        self
    }

    /// Validate and build the [`EmbeddingRequest`]. Returns an error if no inputs were added.
    pub fn build(self) -> anyhow::Result<EmbeddingRequest> {
        if self.inputs.is_empty() {
            anyhow::bail!("Embedding request must contain at least one input.");
        }

        Ok(EmbeddingRequest {
            inputs: self.inputs,
            truncate_sequence: self.truncate_sequence,
        })
    }
}
