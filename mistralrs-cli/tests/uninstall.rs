#![cfg(unix)]

use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use tempfile::TempDir;

const SOURCE_LINE: &str = ". \"$HOME/.mistralrs/env\"\n";

struct Install {
    home: TempDir,
    binary: PathBuf,
}

impl Install {
    fn new() -> Self {
        let home = tempfile::tempdir().unwrap();
        let managed = home.path().join(".mistralrs");
        std::fs::create_dir(&managed).unwrap();
        let binary = managed.join("mistralrs");
        std::fs::copy(env!("CARGO_BIN_EXE_mistralrs"), &binary).unwrap();
        std::fs::write(managed.join("env"), "export MISTRALRS_TEST=1\n").unwrap();
        let bin = home.path().join(".local/bin");
        std::fs::create_dir_all(&bin).unwrap();
        symlink(&binary, bin.join("mistralrs")).unwrap();
        Self { home, binary }
    }

    fn path(&self, path: &str) -> PathBuf {
        self.home.path().join(path)
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.binary);
        command
            .arg("uninstall")
            .env("HOME", self.home.path())
            .env_remove("ZDOTDIR");
        command
    }

    fn uninstall(&self, zdotdir: Option<&Path>) -> Output {
        let mut command = self.command();
        command.arg("--yes");
        if let Some(path) = zdotdir {
            command.env("ZDOTDIR", path);
        }
        command.output().unwrap()
    }
}

#[test]
fn removes_legacy_shell_entries_without_touching_other_content() {
    let install = Install::new();
    let preserved =
        "# user settings\n. \"$HOME/.other/env\"\n# . \"$HOME/.mistralrs/env\"\nexport KEEP=1";
    for rc in [
        ".profile",
        ".bash_profile",
        ".bash_login",
        ".bashrc",
        ".zshenv",
    ] {
        std::fs::write(
            install.path(rc),
            format!("{SOURCE_LINE}{SOURCE_LINE}{preserved}"),
        )
        .unwrap();
    }
    let output = install.uninstall(None);
    assert!(output.status.success(), "{output:?}");
    for rc in [
        ".profile",
        ".bash_profile",
        ".bash_login",
        ".bashrc",
        ".zshenv",
    ] {
        assert_eq!(
            std::fs::read_to_string(install.path(rc)).unwrap(),
            preserved
        );
    }
    assert!(!install.path(".mistralrs").exists());
    assert!(std::fs::symlink_metadata(install.path(".local/bin/mistralrs")).is_err());
}

#[test]
fn removes_entries_from_recorded_and_current_zdotdir() {
    let install = Install::new();
    let old = install.path("old zsh/.zshenv");
    let current = install.path("current zsh/.zshenv");
    for rc in [&old, &current] {
        std::fs::create_dir_all(rc.parent().unwrap()).unwrap();
        std::fs::write(rc, SOURCE_LINE).unwrap();
    }
    std::fs::write(
        install.path(".mistralrs/shell-rcs"),
        format!(
            "{}\n{}\n",
            old.display(),
            install.path("removed-rc").display()
        ),
    )
    .unwrap();
    assert!(install.uninstall(current.parent()).status.success());
    for rc in [&old, &current] {
        assert_eq!(std::fs::read(rc).unwrap(), b"");
    }
    assert!(!install.path("removed-rc").exists());
}

#[test]
fn preserves_symlinks_permissions_and_non_utf8_content() {
    let install = Install::new();
    let target = install.path("bashrc-target");
    let preserved = b"# user bytes: \xff\r\nexport KEEP=1";
    let mut content = SOURCE_LINE.as_bytes().to_vec();
    content.extend_from_slice(preserved);
    std::fs::write(&target, content).unwrap();
    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o640)).unwrap();
    symlink(&target, install.path(".bashrc")).unwrap();
    assert!(install.uninstall(None).status.success());
    assert_eq!(std::fs::read(&target).unwrap(), preserved);
    assert!(install
        .path(".bashrc")
        .symlink_metadata()
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        target.metadata().unwrap().permissions().mode() & 0o777,
        0o640
    );
}

#[test]
fn leaves_unrelated_rc_file_unchanged_and_does_not_create_missing_files() {
    let install = Install::new();
    let rc = install.path(".profile");
    std::fs::write(&rc, "export KEEP=1\n").unwrap();
    let before = rc.metadata().unwrap().modified().unwrap();
    assert!(install.uninstall(None).status.success());
    assert_eq!(rc.metadata().unwrap().modified().unwrap(), before);
    assert_eq!(std::fs::read_to_string(&rc).unwrap(), "export KEEP=1\n");
    assert!(!install.path(".bashrc").exists());
}

#[test]
fn cleanup_failure_keeps_installation_available() {
    let install = Install::new();
    std::fs::create_dir(install.path(".profile")).unwrap();
    let output = install.uninstall(None);
    assert!(!output.status.success());
    assert!(install.binary.exists());
    assert!(install.path(".mistralrs/env").exists());
    assert!(install.path(".local/bin/mistralrs").exists());
}

#[test]
fn cancelled_uninstall_preserves_shell_setup() {
    let install = Install::new();
    std::fs::write(install.path(".profile"), SOURCE_LINE).unwrap();
    let output = install.command().stdin(Stdio::null()).output().unwrap();
    assert!(output.status.success());
    assert!(install.binary.exists());
    assert_eq!(
        std::fs::read_to_string(install.path(".profile")).unwrap(),
        SOURCE_LINE
    );
}

#[test]
fn unmanaged_binary_does_not_modify_shell_setup() {
    let install = Install::new();
    std::fs::write(install.path(".profile"), SOURCE_LINE).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mistralrs"))
        .args(["uninstall", "--yes"])
        .env("HOME", install.home.path())
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(install.binary.exists());
    assert_eq!(
        std::fs::read_to_string(install.path(".profile")).unwrap(),
        SOURCE_LINE
    );
}
