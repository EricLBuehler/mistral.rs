//! Enum types for the OpenResponses API.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Status of a response in the OpenResponses API
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response is queued for processing
    #[default]
    Queued,
    /// Response is currently being processed
    InProgress,
    /// Response completed successfully
    Completed,
    /// Response failed with an error
    Failed,
    /// Response was incomplete (e.g., max tokens reached)
    Incomplete,
    /// Response was cancelled
    Cancelled,
}

impl ResponseStatus {
    /// Returns the string representation of the status
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseStatus::Queued => "queued",
            ResponseStatus::InProgress => "in_progress",
            ResponseStatus::Completed => "completed",
            ResponseStatus::Failed => "failed",
            ResponseStatus::Incomplete => "incomplete",
            ResponseStatus::Cancelled => "cancelled",
        }
    }
}

/// Status of an individual output item
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    /// Item is in progress
    #[default]
    InProgress,
    /// Item completed successfully
    Completed,
    /// Item was incomplete
    Incomplete,
}

impl ItemStatus {
    /// Returns the string representation of the status
    pub fn as_str(&self) -> &'static str {
        match self {
            ItemStatus::InProgress => "in_progress",
            ItemStatus::Completed => "completed",
            ItemStatus::Incomplete => "incomplete",
        }
    }
}

/// Reasoning effort level for models that support extended thinking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No reasoning
    None,
    /// Low reasoning effort
    Low,
    /// Medium reasoning effort (default)
    #[default]
    Medium,
    /// High reasoning effort
    High,
}

/// Truncation strategy for input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum TruncationStrategy {
    /// Automatically truncate input if needed
    #[default]
    Auto,
    /// Disable truncation (may error if input is too long)
    Disabled,
}

/// Service tier for request prioritization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// Automatic tier selection
    #[default]
    Auto,
    /// Default tier
    Default,
    /// Flex tier (lower priority, lower cost)
    Flex,
    /// Priority tier (higher priority)
    Priority,
}

/// Role of a message in the conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// User message
    #[default]
    User,
    /// Assistant message
    Assistant,
    /// System message
    System,
    /// Developer message (similar to system but lower priority)
    Developer,
    /// Tool message (response from a tool call)
    Tool,
}

impl MessageRole {
    /// Returns the string representation of the role
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
            MessageRole::Developer => "developer",
            MessageRole::Tool => "tool",
        }
    }

    /// Parse a string into a MessageRole
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "user" => Some(MessageRole::User),
            "assistant" => Some(MessageRole::Assistant),
            "system" => Some(MessageRole::System),
            "developer" => Some(MessageRole::Developer),
            "tool" => Some(MessageRole::Tool),
            _ => None,
        }
    }
}

/// Image detail level for vision inputs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Automatic detail level
    #[default]
    Auto,
    /// Low detail (faster, less accurate)
    Low,
    /// High detail (slower, more accurate)
    High,
}

/// Reason for incomplete response
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum IncompleteReason {
    /// Maximum tokens reached
    MaxOutputTokens,
    /// Content was filtered
    ContentFilter,
    /// Response was interrupted
    Interrupted,
}

impl IncompleteReason {
    /// Returns the string representation of the reason
    pub fn as_str(&self) -> &'static str {
        match self {
            IncompleteReason::MaxOutputTokens => "max_output_tokens",
            IncompleteReason::ContentFilter => "content_filter",
            IncompleteReason::Interrupted => "interrupted",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_status_as_str() {
        assert_eq!(ResponseStatus::Queued.as_str(), "queued");
        assert_eq!(ResponseStatus::InProgress.as_str(), "in_progress");
        assert_eq!(ResponseStatus::Completed.as_str(), "completed");
        assert_eq!(ResponseStatus::Failed.as_str(), "failed");
        assert_eq!(ResponseStatus::Incomplete.as_str(), "incomplete");
        assert_eq!(ResponseStatus::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn test_response_status_default() {
        assert_eq!(ResponseStatus::default(), ResponseStatus::Queued);
    }

    #[test]
    fn test_response_status_serialization() {
        let status = ResponseStatus::InProgress;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"in_progress\"");

        let deserialized: ResponseStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, status);
    }

    #[test]
    fn test_item_status_as_str() {
        assert_eq!(ItemStatus::InProgress.as_str(), "in_progress");
        assert_eq!(ItemStatus::Completed.as_str(), "completed");
        assert_eq!(ItemStatus::Incomplete.as_str(), "incomplete");
    }

    #[test]
    fn test_item_status_default() {
        assert_eq!(ItemStatus::default(), ItemStatus::InProgress);
    }

    #[test]
    fn test_message_role_as_str() {
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::Developer.as_str(), "developer");
        assert_eq!(MessageRole::Tool.as_str(), "tool");
    }

    #[test]
    fn test_message_role_from_str() {
        assert_eq!(MessageRole::parse("user"), Some(MessageRole::User));
        assert_eq!(
            MessageRole::parse("assistant"),
            Some(MessageRole::Assistant)
        );
        assert_eq!(MessageRole::parse("system"), Some(MessageRole::System));
        assert_eq!(
            MessageRole::parse("developer"),
            Some(MessageRole::Developer)
        );
        assert_eq!(MessageRole::parse("tool"), Some(MessageRole::Tool));
        assert_eq!(MessageRole::parse("USER"), Some(MessageRole::User)); // case insensitive
        assert_eq!(MessageRole::parse("invalid"), None);
    }

    #[test]
    fn test_message_role_default() {
        assert_eq!(MessageRole::default(), MessageRole::User);
    }

    #[test]
    fn test_incomplete_reason_as_str() {
        assert_eq!(
            IncompleteReason::MaxOutputTokens.as_str(),
            "max_output_tokens"
        );
        assert_eq!(IncompleteReason::ContentFilter.as_str(), "content_filter");
        assert_eq!(IncompleteReason::Interrupted.as_str(), "interrupted");
    }

    #[test]
    fn test_incomplete_reason_serialization() {
        let reason = IncompleteReason::MaxOutputTokens;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"max_output_tokens\"");

        let deserialized: IncompleteReason = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, reason);
    }

    #[test]
    fn test_reasoning_effort_default() {
        assert_eq!(ReasoningEffort::default(), ReasoningEffort::Medium);
    }

    #[test]
    fn test_truncation_strategy_default() {
        assert_eq!(TruncationStrategy::default(), TruncationStrategy::Auto);
    }

    #[test]
    fn test_service_tier_default() {
        assert_eq!(ServiceTier::default(), ServiceTier::Auto);
    }

    #[test]
    fn test_image_detail_default() {
        assert_eq!(ImageDetail::default(), ImageDetail::Auto);
    }
}
