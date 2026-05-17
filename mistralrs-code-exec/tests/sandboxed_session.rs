#![cfg(target_os = "linux")]

use std::path::PathBuf;

use mistralrs_code_exec::{CodeExecutionConfig, CodeExecutionManager};
use mistralrs_mcp::{CalledFunction, ToolCallContext, ToolCallbackKind};
use mistralrs_sandbox::{NetworkMode, SandboxPolicy};

fn which(prog: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(prog))
            .find(|p| p.is_file())
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sandboxed_session_can_execute_python() {
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        sandbox_policy: Some(SandboxPolicy {
            network: NetworkMode::None,
            ..SandboxPolicy::default()
        }),
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct under sandbox");

    let callbacks = manager.get_tool_callbacks(&[]);
    let exec = callbacks
        .get(mistralrs_code_exec::EXECUTE_PYTHON_TOOL_NAME)
        .expect("execute_python tool registered");

    let func = CalledFunction {
        name: mistralrs_code_exec::EXECUTE_PYTHON_TOOL_NAME.to_string(),
        arguments: r#"{"code": "print(2 + 2)"}"#.to_string(),
    };
    let ctx = ToolCallContext {
        session_id: Some("test".to_string()),
    };

    let result = match &exec.callback {
        ToolCallbackKind::Multimodal(cb) => cb(&func, &ctx).expect("tool call should succeed"),
        _ => panic!("execute_python should be Multimodal"),
    };

    let json = match result {
        mistralrs_mcp::ToolOutput::Multimodal { text, .. } => text,
        mistralrs_mcp::ToolOutput::Text(t) => t,
    };

    assert!(
        json.contains("\"4\\n\"") || json.contains("4\\n"),
        "expected stdout '4' in tool output; got: {json}"
    );
}
