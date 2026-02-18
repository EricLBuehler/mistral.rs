/// One-line text generation. Loads a model with Q4K ISQ quantization,
/// sends the prompt, and returns the model's text reply.
///
/// This is a convenience wrapper intended for quick prototyping and scripts.
/// For production usage, build a [`Model`](crate::Model) explicitly via
/// [`TextModelBuilder`](crate::TextModelBuilder) to reuse it across requests.
///
/// # Example
/// ```no_run
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let answer = mistralrs::generate("microsoft/Phi-3.5-mini-instruct", "What is 2+2?").await?;
///     println!("{answer}");
///     Ok(())
/// }
/// ```
pub async fn generate(model_id: &str, prompt: &str) -> crate::error::Result<String> {
    let model = crate::TextModelBuilder::new(model_id)
        .with_isq(crate::IsqType::Q4K)
        .build()
        .await
        .map_err(|e| crate::error::Error::ModelLoad(e.into()))?;
    model.chat(prompt).await
}
