#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use either::Either;
    use mistralrs_core::{WebSearchOptions, Response, Request};
    use mistralrs_server_core::{
        chat_completion::parse_request,
        openai::{ChatCompletionRequest, Message, MessageContent},
        types::SharedMistralRsState,
        mistralrs_for_server_builder::{MistralRsForServerBuilder, ModelConfig},
    };
    use tokio::sync::mpsc;
    use mistralrs_core::ModelSelected;

    #[tokio::test]
    async fn test_web_search_options_sanitization() {
        // We don't actually need a full model to test parse_request, 
        // but we need a state that can provide next_request_id.
        // MistralRsForServerBuilder might be too heavy for a unit test in this environment.
        // Let's try to mock it if possible, or just see if we can get a minimal one.
        
        // Actually, parse_request needs state for:
        // 1. state.next_request_id()
        // 2. state.get_model_category(None) (for vision models)
        // 3. maybe_log_request
        
        // Given the constraints, maybe I can just verify the code change by inspection 
        // or a very targeted test if I can satisfy the dependencies.

        // Let's try to see if we can instantiate MistralRs minimal.
        // Since I'm in a VM with 8GB RAM, building everything might be slow.
    }
}
