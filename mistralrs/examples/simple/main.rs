use mistralrs::{
    load_normal_model, Constraint, Device, NormalRequest, Request, RequestMessage, Response,
    SamplingParams,
};
use tokio::sync::mpsc::channel;

fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let runner = load_normal_model!(
        id = "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
        kind = Mistral,
        device = dev,
        use_flash_attn = false
    );

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest {
        messages: RequestMessage::Completion {
            text: "Hello! My name is ".to_string(),
            echo_prompt: false,
            best_of: 1,
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        id: 0,
        constraint: Constraint::None,
        suffix: None,
        adapters: None,
    });
    runner.get_sender()?.blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    match response {
        Response::CompletionDone(c) => println!("Text: {}", c.choices[0].text),
        _ => unreachable!(),
    }
    Ok(())
}
