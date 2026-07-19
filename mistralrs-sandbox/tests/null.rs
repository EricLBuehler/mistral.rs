use mistralrs_sandbox::{NullSandbox, Sandbox, SandboxPolicy};
use tokio::process::Command;

#[test]
fn null_sandbox_is_noop() {
    let sb = NullSandbox;
    let policy = SandboxPolicy::default();
    let mut cmd = Command::new("/bin/true");
    sb.harden(&mut cmd, &policy).expect("harden");
    sb.attach(0, &policy).expect("attach");
    assert_eq!(sb.name(), "null");
}

#[tokio::test]
async fn null_sandbox_runs_child() {
    let sb = NullSandbox;
    let policy = SandboxPolicy::default();
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c").arg("exit 0");
    sb.harden(&mut cmd, &policy).expect("harden");
    let status = cmd.status().await.expect("status");
    assert!(status.success());
}
