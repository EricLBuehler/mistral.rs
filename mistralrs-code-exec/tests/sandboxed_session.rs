#![cfg(target_os = "linux")]

use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use mistralrs_code_exec::{
    CodeExecutionApproval, CodeExecutionConfig, CodeExecutionManager, CodeExecutionPermission,
};
use mistralrs_mcp::{
    CalledFunction, CodeExecutionPermission as RequestCodeExecutionPermission, ToolCallContext,
    ToolCallbackKind,
};
use mistralrs_sandbox::{NetworkMode, SandboxPolicy};

static TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

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
    exec_json_with_permission(callbacks, session_id, code, None)
}

fn exec_json_with_permission(
    callbacks: &std::collections::HashMap<String, mistralrs_mcp::ToolCallbackWithTool>,
    session_id: &str,
    code: &str,
    code_execution_permission: Option<RequestCodeExecutionPermission>,
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
        round: None,
        tool_name: None,
        code_execution_permission,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_notifier: None,
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

fn reset_json(
    callbacks: &std::collections::HashMap<String, mistralrs_mcp::ToolCallbackWithTool>,
    session_id: &str,
) -> String {
    let reset = callbacks
        .get(mistralrs_code_exec::RESET_SESSION_TOOL_NAME)
        .expect("reset_python_session tool registered");

    let func = CalledFunction {
        name: mistralrs_code_exec::RESET_SESSION_TOOL_NAME.to_string(),
        arguments: "{}".to_string(),
    };
    let ctx = ToolCallContext {
        session_id: Some(session_id.to_string()),
        round: None,
        tool_name: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_notifier: None,
    };

    match &reset.callback {
        ToolCallbackKind::Text(cb) => cb(&func, &ctx).expect("tool call should succeed"),
        _ => panic!("reset_python_session should be Text"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sandboxed_session_can_execute_python() {
    let _guard = TEST_LOCK.lock().await;
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
    let _guard = TEST_LOCK.lock().await;
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
    let _guard = TEST_LOCK.lock().await;
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
    let _guard = TEST_LOCK.lock().await;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn code_execution_permission_deny_blocks_python() {
    let _guard = TEST_LOCK.lock().await;
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        permission: CodeExecutionPermission::Deny,
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct");
    let callbacks = manager.get_tool_callbacks(&[]);

    let json = exec_json(
        &callbacks,
        "deny-test",
        "raise RuntimeError('should not run')",
    );

    assert!(json.contains("\"status\":\"denied\""), "{json}");
    assert!(json.contains("denied by policy"), "{json}");
    assert!(!json.contains("should not run"), "{json}");

    let json = reset_json(&callbacks, "deny-test");
    assert!(json.contains("\"status\":\"denied\""), "{json}");
    assert!(json.contains("denied by policy"), "{json}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_permission_can_tighten_to_deny() {
    let _guard = TEST_LOCK.lock().await;
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        permission: CodeExecutionPermission::Auto,
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct");
    let callbacks = manager.get_tool_callbacks(&[]);

    let json = exec_json_with_permission(
        &callbacks,
        "request-deny-test",
        "raise RuntimeError('should not run')",
        Some(RequestCodeExecutionPermission::Deny),
    );

    assert!(json.contains("\"status\":\"denied\""), "{json}");
    assert!(json.contains("denied by policy"), "{json}");
    assert!(!json.contains("should not run"), "{json}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_permission_cannot_loosen_global_policy() {
    let _guard = TEST_LOCK.lock().await;
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        permission: CodeExecutionPermission::Deny,
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct");
    let callbacks = manager.get_tool_callbacks(&[]);

    let json = exec_json_with_permission(
        &callbacks,
        "global-deny-test",
        "print('should not run')",
        Some(RequestCodeExecutionPermission::Auto),
    );

    assert!(json.contains("\"status\":\"denied\""), "{json}");
    assert!(json.contains("denied by policy"), "{json}");
    assert!(!json.contains("should not run"), "{json}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn code_execution_permission_ask_uses_callback() {
    let _guard = TEST_LOCK.lock().await;
    let Some(python) = which("python3") else {
        eprintln!("skipping: python3 not on PATH");
        return;
    };

    let cfg = CodeExecutionConfig {
        python_path: python,
        permission: CodeExecutionPermission::Ask,
        approval_callback: Some(Arc::new(|approval: &CodeExecutionApproval| {
            approval.code.contains("print")
        })),
        ..CodeExecutionConfig::default()
    };

    let manager = CodeExecutionManager::new(cfg)
        .await
        .expect("manager should construct");
    let callbacks = manager.get_tool_callbacks(&[]);

    let json = exec_json(&callbacks, "ask-test", "print(8 + 9)");

    assert!(
        json.contains("\"17\\n\"") || json.contains("17\\n"),
        "{json}"
    );
}
