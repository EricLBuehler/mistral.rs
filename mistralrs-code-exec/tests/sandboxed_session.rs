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

fn exec_json(
    callbacks: &std::collections::HashMap<String, mistralrs_mcp::ToolCallbackWithTool>,
    session_id: &str,
    code: &str,
) -> String {
    let exec = callbacks
        .get(mistralrs_code_exec::EXECUTE_PYTHON_TOOL_NAME)
        .expect("execute_python tool registered");

    let func = CalledFunction {
        name: mistralrs_code_exec::EXECUTE_PYTHON_TOOL_NAME.to_string(),
        arguments: serde_json::json!({ "code": code }).to_string(),
    };
    let ctx = ToolCallContext {
        session_id: Some(session_id.to_string()),
    };

    let result = match &exec.callback {
        ToolCallbackKind::Multimodal(cb) => cb(&func, &ctx).expect("tool call should succeed"),
        _ => panic!("execute_python should be Multimodal"),
    };

    match result {
        mistralrs_mcp::ToolOutput::Multimodal { text, .. } => text,
        mistralrs_mcp::ToolOutput::Text(t) => t,
    }
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
    let json = exec_json(&callbacks, "test", "print(2 + 2)");

    assert!(
        json.contains("\"4\\n\"") || json.contains("4\\n"),
        "expected stdout '4' in tool output; got: {json}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sandboxed_session_default_policy_can_execute_python() {
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        sandbox_policy: Some(SandboxPolicy::default()),
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct under default sandbox");

    let callbacks = manager.get_tool_callbacks(&[]);
    let json = exec_json(&callbacks, "default-policy-test", "print(2 + 3)");

    assert!(
        json.contains("\"5\\n\"") || json.contains("5\\n"),
        "expected stdout '5' in tool output; got: {json}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn callbacks_outlive_manager_executor_tempdir() {
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        sandbox_policy: Some(SandboxPolicy::default()),
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct under default sandbox");
    let callbacks = manager.get_tool_callbacks(&[]);
    drop(manager);

    let json = exec_json(&callbacks, "dropped-manager-test", "print(6 + 7)");

    assert!(
        json.contains("\"13\\n\"") || json.contains("13\\n"),
        "expected stdout '13' in tool output; got: {json}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subprocess_eof_reports_recent_stderr() {
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct");
    let callbacks = manager.get_tool_callbacks(&[]);

    let json = exec_json(
        &callbacks,
        "stderr-eof-test",
        "import os, sys\nprint('child boom', file=sys.__stderr__, flush=True)\nos._exit(7)",
    );

    assert!(json.contains("Python subprocess closed stdout"), "{json}");
    assert!(json.contains("child boom"), "{json}");
}
