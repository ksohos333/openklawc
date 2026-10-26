//! Project context injection — auto-detect project type and inject metadata.
//!
//! Detects projects by looking for marker files (Cargo.toml, package.json, etc.)
//! and reads package lists, build configs, and git history as context.

use std::path::{Path, PathBuf};

use anyhow::Result;
use tracing::{debug, warn};

/// Detected project type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ProjectType {
    #[default]
    Unknown,
    Rust,
    Node,
    Python,
    Go,
    Java,
}

/// Parsed context for a discovered project.
#[derive(Debug, Clone, Default)]
pub struct ProjectContext {
    pub project_type: ProjectType,
    pub root: PathBuf,
    pub name: Option<String>,
    pub version: Option<String>,
    pub dependencies: Vec<String>,
    pub git_summary: Option<GitSummary>,
    pub file_count: usize,
    pub recent_files: Vec<String>,
}

/// Git history summary.
#[derive(Debug, Clone)]
pub struct GitSummary {
    pub branch: String,
    pub recent_commits: Vec<String>,
    pub files_changed: Vec<String>,
    pub has_uncommitted: bool,
}

/// Detect project type and build context from a directory.
pub fn detect_project(dir: &Path) -> Result<ProjectContext> {
    let mut ctx = ProjectContext {
        root: dir.to_path_buf(),
        ..Default::default()
    };

    // Detect by marker files
    if dir.join("Cargo.toml").exists() {
        ctx.project_type = ProjectType::Rust;
        if let Err(e) = parse_cargo_toml(&mut ctx, &dir.join("Cargo.toml")) {
            warn!("Failed to parse Cargo.toml: {}", e);
        }
    } else if dir.join("package.json").exists() {
        ctx.project_type = ProjectType::Node;
        if let Err(e) = parse_package_json(&mut ctx, &dir.join("package.json")) {
            warn!("Failed to parse package.json: {}", e);
        }
    } else if dir.join("pyproject.toml").exists() || dir.join("requirements.txt").exists() {
        ctx.project_type = ProjectType::Python;
        if let Err(e) = parse_python_project(&mut ctx, dir) {
            warn!("Failed to parse Python project: {}", e);
        }
    } else if dir.join("go.mod").exists() {
        ctx.project_type = ProjectType::Go;
    } else if dir.join("pom.xml").exists() || dir.join("build.gradle").exists() {
        ctx.project_type = ProjectType::Java;
    } else {
        ctx.project_type = ProjectType::Unknown;
    }

    // Gather file statistics
    ctx.file_count = count_source_files(dir, &ctx.project_type);
    ctx.recent_files = recent_source_files(dir, 10, &ctx.project_type);

    // Git info
    ctx.git_summary = get_git_summary(dir).ok();

    debug!("Project context: {:?}", ctx);
    Ok(ctx)
}

/// Format the context as a short string for system prompts.
pub fn format_context(ctx: &ProjectContext) -> String {
    let mut lines = Vec::new();

    let type_name = match ctx.project_type {
        ProjectType::Rust => "Rust",
        ProjectType::Node => "Node.js",
        ProjectType::Python => "Python",
        ProjectType::Go => "Go",
        ProjectType::Java => "Java",
        ProjectType::Unknown => "Unknown",
    };

    lines.push(format!("Project type: {}", type_name));

    if let Some(name) = &ctx.name {
        lines.push(format!("Project name: {}", name));
    }
    if let Some(version) = &ctx.version {
        lines.push(format!("Version: {}", version));
    }

    lines.push(format!("Source files: {}", ctx.file_count));

    if !ctx.dependencies.is_empty() {
        lines.push(format!(
            "Dependencies ({}): {}",
            ctx.dependencies.len(),
            if ctx.dependencies.len() <= 10 {
                ctx.dependencies.join(", ")
            } else {
                let first: Vec<_> = ctx.dependencies.iter().take(10).cloned().collect();
                format!(
                    "{}, ... +{} more",
                    first.join(", "),
                    ctx.dependencies.len() - 10
                )
            }
        ));
    }

    if let Some(git) = &ctx.git_summary {
        lines.push(format!("Git branch: {}", git.branch));
        if git.has_uncommitted {
            lines.push("Git status: uncommitted changes present".to_string());
        }
        if !git.recent_commits.is_empty() {
            lines.push(format!(
                "Recent commits:\n  {}",
                git.recent_commits.join("\n  ")
            ));
        }
    }

    if !ctx.recent_files.is_empty() {
        lines.push(format!(
            "Recently active files:\n  {}",
            ctx.recent_files.join("\n  ")
        ));
    }

    lines.join("\n")
}

fn parse_cargo_toml(ctx: &mut ProjectContext, path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let parsed: toml::Value = toml::from_str(&text)?;

    ctx.name = parsed
        .get("package")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .map(String::from);

    ctx.version = parsed
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str())
        .map(String::from);

    // Extract workspace members
    if let Some(members) = parsed
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
    {
        for member in members {
            if let Some(s) = member.as_str() {
                ctx.dependencies.push(format!("workspace: {}", s));
            }
        }
    }

    // Extract dependencies
    for table in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(deps) = parsed.get(table).and_then(|d| d.as_table()) {
            for (name, _) in deps {
                ctx.dependencies.push(name.clone());
            }
        }
    }

    Ok(())
}

fn parse_package_json(ctx: &mut ProjectContext, path: &Path) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let parsed: serde_json::Value = serde_json::from_str(&text)?;

    ctx.name = parsed
        .get("name")
        .and_then(|n| n.as_str())
        .map(String::from);

    ctx.version = parsed
        .get("version")
        .and_then(|v| v.as_str())
        .map(String::from);

    for table in ["dependencies", "devDependencies", "peerDependencies"] {
        if let Some(deps) = parsed.get(table).and_then(|d| d.as_object()) {
            for (name, _) in deps {
                ctx.dependencies.push(name.clone());
            }
        }
    }

    Ok(())
}

fn parse_python_project(ctx: &mut ProjectContext, dir: &Path) -> Result<()> {
    if let Ok(text) = std::fs::read_to_string(dir.join("pyproject.toml")) {
        let parsed: toml::Value = toml::from_str(&text)?;
        if let Some(deps) = parsed
            .get("project")
            .and_then(|p| p.get("dependencies"))
            .and_then(|d| d.as_array())
        {
            for dep in deps {
                if let Some(s) = dep.as_str() {
                    // Extract just the package name (strip version specifiers)
                    let name = s
                        .split(['=', '<', '>', ';'])
                        .next()
                        .unwrap_or(s)
                        .trim()
                        .to_string();
                    if !name.is_empty() {
                        ctx.dependencies.push(name);
                    }
                }
            }
        }
    }

    // Fallback: read requirements.txt
    if let Ok(text) = std::fs::read_to_string(dir.join("requirements.txt")) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let name = line
                .split(['=', '<', '>', '['])
                .next()
                .unwrap_or(line)
                .trim()
                .to_string();
            if !name.is_empty() && !ctx.dependencies.contains(&name) {
                ctx.dependencies.push(name);
            }
        }
    }

    Ok(())
}

fn source_extensions(ty: &ProjectType) -> &[&str] {
    match ty {
        ProjectType::Rust => &["rs", "toml"],
        ProjectType::Node => &["js", "ts", "jsx", "tsx", "json", "mjs"],
        ProjectType::Python => &["py", "pyi", "ipynb"],
        ProjectType::Go => &["go", "mod"],
        ProjectType::Java => &["java", "kt", "gradle"],
        ProjectType::Unknown => &[],
    }
}

fn count_source_files(dir: &Path, ty: &ProjectType) -> usize {
    let exts = source_extensions(ty);
    let mut count = 0;
    for e in walkdir::WalkDir::new(dir)
        .max_depth(2)
        .into_iter()
        .flatten()
    {
        if e.file_type().is_file() {
            if let Some(ext) = e.path().extension().and_then(|e| e.to_str()) {
                if exts.contains(&ext) {
                    count += 1;
                }
            }
        }
    }
    count
}

fn recent_source_files(dir: &Path, limit: usize, ty: &ProjectType) -> Vec<String> {
    let exts = source_extensions(ty);
    let mut files: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

    for e in walkdir::WalkDir::new(dir)
        .max_depth(3)
        .into_iter()
        .flatten()
    {
        if e.file_type().is_file() {
            if let Some(ext) = e.path().extension().and_then(|e| e.to_str()) {
                if exts.contains(&ext) {
                    if let Ok(meta) = e.metadata() {
                        if let Ok(modified) = meta.modified() {
                            files.push((e.path().to_path_buf(), modified));
                        }
                    }
                }
            }
        }
    }

    files.sort_by(|a, b| b.1.cmp(&a.1));
    files
        .into_iter()
        .take(limit)
        .map(|(p, _)| {
            p.strip_prefix(dir)
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string()
        })
        .collect()
}

fn get_git_summary(dir: &Path) -> Result<GitSummary> {
    use std::process::Command;

    let branch = Command::new("git")
        .args(["-C", &dir.to_string_lossy(), "branch", "--show-current"])
        .output()?;
    let branch = String::from_utf8_lossy(&branch.stdout).trim().to_string();

    let log = Command::new("git")
        .args([
            "-C",
            &dir.to_string_lossy(),
            "log",
            "-3",
            "--oneline",
            "--no-decorate",
        ])
        .output()?;
    let commits = String::from_utf8_lossy(&log.stdout)
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let status = Command::new("git")
        .args(["-C", &dir.to_string_lossy(), "status", "--short"])
        .output()?;
    let changed: Vec<String> = String::from_utf8_lossy(&status.stdout)
        .lines()
        .map(|l| l[3..].trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let has_uncommitted = !changed.is_empty();

    Ok(GitSummary {
        branch,
        recent_commits: commits,
        files_changed: changed,
        has_uncommitted,
    })
}
