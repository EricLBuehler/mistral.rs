use std::env;
use std::path::PathBuf;
use std::process::Command;

pub const DEFAULT_METAL_STD: &str = "metal3.1";

const PRECOMPILE_ENV: &str = "MISTRALRS_METAL_PRECOMPILE";
const PLATFORMS_ENV: &str = "MISTRALRS_METAL_PLATFORMS";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalPlatform {
    MacOS,
    Ios,
    TvOS,
}

impl MetalPlatform {
    const ALL: [Self; 3] = [Self::MacOS, Self::Ios, Self::TvOS];

    fn sdk(self) -> &'static str {
        match self {
            Self::MacOS => "macosx",
            Self::Ios => "iphoneos",
            Self::TvOS => "appletvos",
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::MacOS => "",
            Self::Ios => "_ios",
            Self::TvOS => "_tvos",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "macos" | "macosx" => Some(Self::MacOS),
            "ios" | "iphoneos" => Some(Self::Ios),
            "tvos" | "appletvos" => Some(Self::TvOS),
            _ => None,
        }
    }
}

pub struct MetalBuildConfig<'a> {
    pub library_name: &'a str,
    pub source_dir: &'a str,
    pub metal_sources: &'a [&'a str],
    pub header_sources: &'a [&'a str],
    pub include_only_sources: &'a [&'a str],
    pub metal_std: &'a str,
}

pub fn compile(config: &MetalBuildConfig<'_>) -> Result<(), String> {
    emit_rerun_directives(config);
    if should_skip_precompile() {
        println!("cargo:warning=Skipping Metal kernel precompilation ({PRECOMPILE_ENV}=0)");
        write_dummy_metallibs(config)?;
        return Ok(());
    }

    let platforms = selected_platforms()?;
    ensure_target_platform_selected(&platforms)?;
    for platform in platforms {
        compile_platform(config, platform)?;
    }
    Ok(())
}

fn emit_rerun_directives(config: &MetalBuildConfig<'_>) {
    for source in config
        .metal_sources
        .iter()
        .chain(config.header_sources)
        .chain(config.include_only_sources)
    {
        println!(
            "cargo::rerun-if-changed={}/{}.metal",
            config.source_dir, source
        );
    }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={PRECOMPILE_ENV}");
    println!("cargo:rerun-if-env-changed={PLATFORMS_ENV}");
}

fn should_skip_precompile() -> bool {
    env::var(PRECOMPILE_ENV)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn selected_platforms() -> Result<Vec<MetalPlatform>, String> {
    let raw = match env::var(PLATFORMS_ENV) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(MetalPlatform::ALL.to_vec()),
        Err(err) => return Err(format!("Failed to read {PLATFORMS_ENV}: {err}")),
    };

    let mut platforms = Vec::new();
    for token in raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if token.eq_ignore_ascii_case("all") {
            return Ok(MetalPlatform::ALL.to_vec());
        }
        let platform = MetalPlatform::parse(token).ok_or_else(|| {
            format!("Invalid {PLATFORMS_ENV} entry `{token}`; expected macos, ios, tvos, or all")
        })?;
        if !platforms.contains(&platform) {
            platforms.push(platform);
        }
    }

    if platforms.is_empty() {
        Err(format!(
            "{PLATFORMS_ENV} must contain macos, ios, tvos, or all"
        ))
    } else {
        Ok(platforms)
    }
}

fn ensure_target_platform_selected(platforms: &[MetalPlatform]) -> Result<(), String> {
    let Some(target_platform) = target_platform()? else {
        return Ok(());
    };
    if platforms.contains(&target_platform) {
        return Ok(());
    }
    Err(format!(
        "{PLATFORMS_ENV} does not include the current target platform: {:?}",
        target_platform
    ))
}

fn target_platform() -> Result<Option<MetalPlatform>, String> {
    let target_os = match env::var("CARGO_CFG_TARGET_OS") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(err) => return Err(format!("Failed to read CARGO_CFG_TARGET_OS: {err}")),
    };
    Ok(match target_os.as_str() {
        "macos" => Some(MetalPlatform::MacOS),
        "ios" => Some(MetalPlatform::Ios),
        "tvos" => Some(MetalPlatform::TvOS),
        _ => None,
    })
}

fn write_dummy_metallibs(config: &MetalBuildConfig<'_>) -> Result<(), String> {
    let out_dir = out_dir()?;
    for platform in MetalPlatform::ALL {
        std::fs::write(out_dir.join(metallib_name(config, platform)), [])
            .map_err(|err| format!("Failed to write dummy metallib: {err}"))?;
    }
    Ok(())
}

fn compile_platform(config: &MetalBuildConfig<'_>, platform: MetalPlatform) -> Result<(), String> {
    let current_dir = env::current_dir().map_err(|err| format!("Failed to get cwd: {err}"))?;
    let out_dir = out_dir()?;
    let working_directory = out_dir.to_string_lossy().to_string();
    let sources = current_dir.join(config.source_dir);

    let mut compile_air_cmd = Command::new("xcrun");
    compile_air_cmd
        .arg("--sdk")
        .arg(platform.sdk())
        .arg("metal")
        .arg(format!("-std={}", config.metal_std))
        .arg(format!("-working-directory={working_directory}"))
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-O3")
        .arg("-c")
        .arg("-w");
    for metal_file in config.metal_sources.iter().chain(config.header_sources) {
        compile_air_cmd.arg(sources.join(format!("{metal_file}.metal")));
    }
    run_command(&mut compile_air_cmd, "Compiling metal -> air")?;

    let metallib = out_dir.join(metallib_name(config, platform));
    let mut compile_metallib_cmd = Command::new("xcrun");
    compile_metallib_cmd
        .arg("--sdk")
        .arg(platform.sdk())
        .arg("metal")
        .arg(format!("-std={}", config.metal_std))
        .arg("-o")
        .arg(&metallib);
    for metal_file in config.metal_sources.iter().chain(config.header_sources) {
        compile_metallib_cmd.arg(out_dir.join(format!("{metal_file}.air")));
    }
    run_command(&mut compile_metallib_cmd, "Compiling air -> metallib")
}

fn out_dir() -> Result<PathBuf, String> {
    env::var("OUT_DIR")
        .map(PathBuf::from)
        .map_err(|_| "OUT_DIR not set".to_string())
}

fn metallib_name(config: &MetalBuildConfig<'_>, platform: MetalPlatform) -> String {
    format!("{}{}.metallib", config.library_name, platform.suffix())
}

fn run_command(cmd: &mut Command, action: &str) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|err| format!("{action} failed to start: {err}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{action} failed with status: {status}"))
    }
}
