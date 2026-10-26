//! mcp-filesystem — Rust MCP server for sandboxed filesystem access.
//!
//! A Rust port of `@modelcontextprotocol/server-filesystem`. Every path is
//! checked against an allow-list; symlinks are resolved before the check so
//! they cannot escape the sandbox.
//!
//! Tools:
//!   read_file(path, head?, tail?)            — deprecated alias
//!   read_text_file(path, head?, tail?)
//!   read_media_file(path)
//!   read_multiple_files(paths[])
//!   write_file(path, content)
//!   edit_file(path, edits[], dryRun?)
//!   create_directory(path)
//!   list_directory(path)
//!   list_directory_with_sizes(path, sortBy?)
//!   directory_tree(path, excludePatterns?)
//!   move_file(source, destination)
//!   search_files(path, pattern, excludePatterns?)
//!   get_file_info(path)
//!   list_allowed_directories()

mod ops;
mod paths;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use ops::Edit;

type Allowed = Arc<Vec<PathBuf>>;

/// Read the `path` argument, rejecting non-string values.
fn need_path(args: &Value, key: &str) -> Result<String> {
    args[key]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("{} required", key))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let allowed: Allowed = Arc::new(paths::allowed_directories_from_cli());
    tracing::info!(
        "filesystem server allowed directories: {:?}",
        allowed
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
    );

    let mut server = okc_mcp::McpServer::new("secure-filesystem-server");

    // ─── read_text_file ───────────────────────────────────────────────────
    let read_text_schema = json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "tail": { "type": "integer", "description": "Return only the last N lines" },
            "head": { "type": "integer", "description": "Return only the first N lines" }
        },
        "required": ["path"]
    });

    for (tool_name, description) in [
        (
            "read_text_file",
            "Read the complete contents of a file from the file system as text. \
             Use 'head' to read only the first N lines or 'tail' for the last N lines. \
             Only works within allowed directories.",
        ),
        (
            "read_file",
            "Read the complete contents of a file as text. DEPRECATED: use read_text_file instead.",
        ),
    ] {
        let dirs = allowed.clone();
        server.add_tool(
            tool_name,
            description,
            read_text_schema.clone(),
            Box::new(move |args| {
                let allowed = dirs.clone();
                Box::pin(async move {
                    let path = need_path(&args, "path")?;
                    let head = args["head"].as_u64().map(|v| v as usize);
                    let tail = args["tail"].as_u64().map(|v| v as usize);
                    let valid = paths::validate_path(&path, &allowed)?;
                    let content = ops::read_text_file(&valid, head, tail)?;
                    Ok(json!(content))
                })
            }),
        );
    }

    // ─── read_media_file ──────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "read_media_file",
        "Read a file and return it as a base64-encoded content block with its MIME type. \
         Image and audio files are returned as image/audio content; any other file type is \
         returned as an embedded resource. Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let valid = paths::validate_path(&path, &allowed)?;
                let block = ops::read_media_file(&valid)?;
                Ok(json!({ "content": [block] }))
            })
        }),
    );

    // ─── read_multiple_files ──────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "read_multiple_files",
        "Read the contents of multiple files simultaneously. Each file's content is returned \
         with its path as a reference. Failed reads for individual files won't stop the entire \
         operation. Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "paths": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["paths"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let paths = args["paths"]
                    .as_array()
                    .ok_or_else(|| anyhow!("paths required"))?;
                let mut results = Vec::new();
                for entry in paths {
                    let raw = entry
                        .as_str()
                        .ok_or_else(|| anyhow!("paths must be strings"))?;
                    match paths::validate_path(raw, &allowed) {
                        Ok(valid) => match ops::read_text_file(&valid, None, None) {
                            Ok(content) => results.push(format!("{}:\n{}\n", raw, content)),
                            Err(e) => results.push(format!("{}: Error - {}", raw, e)),
                        },
                        Err(e) => results.push(format!("{}: Error - {}", raw, e)),
                    }
                }
                Ok(json!(results.join("\n---\n")))
            })
        }),
    );

    // ─── write_file ───────────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "write_file",
        "Create a new file or completely overwrite an existing file with new content. \
         Use with caution as it will overwrite existing files without warning. \
         Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let content = args["content"]
                    .as_str()
                    .ok_or_else(|| anyhow!("content required"))?;
                let valid = paths::validate_path(&path, &allowed)?;
                ops::write_file(&valid, content)?;
                Ok(json!(format!("Successfully wrote to {}", path)))
            })
        }),
    );

    // ─── edit_file ────────────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "edit_file",
        "Make line-based edits to a text file. Each edit replaces an exact line sequence \
         with new content. Returns a git-style diff showing the changes made. \
         Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string" },
                            "newText": { "type": "string" }
                        },
                        "required": ["oldText", "newText"]
                    }
                },
                "dryRun": { "type": "boolean", "default": false }
            },
            "required": ["path", "edits"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let raw_edits = args["edits"]
                    .as_array()
                    .ok_or_else(|| anyhow!("edits required"))?;
                let mut edits = Vec::new();
                for entry in raw_edits {
                    let old_text = entry["oldText"]
                        .as_str()
                        .ok_or_else(|| anyhow!("edit.oldText required"))?;
                    let new_text = entry["newText"]
                        .as_str()
                        .ok_or_else(|| anyhow!("edit.newText required"))?;
                    edits.push(Edit {
                        old_text: old_text.to_string(),
                        new_text: new_text.to_string(),
                    });
                }
                let dry_run = args["dryRun"].as_bool().unwrap_or(false);
                let valid = paths::validate_path(&path, &allowed)?;
                let diff = ops::edit_file(&valid, &edits, dry_run)?;
                let mut backticks = 3;
                while diff.contains(&"`".repeat(backticks)) {
                    backticks += 1;
                }
                let fence = "`".repeat(backticks);
                Ok(json!(format!("{}diff\n{}{}\n\n", fence, diff, fence)))
            })
        }),
    );

    // ─── create_directory ─────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "create_directory",
        "Create a new directory or ensure a directory exists. Can create multiple nested \
         directories in one operation. If the directory already exists, this operation will \
         succeed silently. Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let valid = paths::validate_path(&path, &allowed)?;
                ops::create_directory(&valid)?;
                Ok(json!(format!("Successfully created directory {}", path)))
            })
        }),
    );

    // ─── list_directory ───────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "list_directory",
        "Get a detailed listing of all files and directories in a specified path. Results \
         clearly distinguish between files and directories with [FILE] and [DIR] prefixes. \
         Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let valid = paths::validate_path(&path, &allowed)?;
                Ok(json!(ops::list_directory(&valid)?))
            })
        }),
    );

    // ─── list_directory_with_sizes ────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "list_directory_with_sizes",
        "Get a detailed listing of all files and directories in a specified path, including \
         sizes. Results clearly distinguish between files and directories with [FILE] and \
         [DIR] prefixes. Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "sortBy": { "type": "string", "enum": ["name", "size"], "default": "name" }
            },
            "required": ["path"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let sort_by_size = args["sortBy"].as_str() == Some("size");
                let valid = paths::validate_path(&path, &allowed)?;
                Ok(json!(ops::list_directory_with_sizes(&valid, sort_by_size)?))
            })
        }),
    );

    // ─── directory_tree ───────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "directory_tree",
        "Get a recursive tree view of files and directories as a JSON structure. Each entry \
         includes 'name', 'type' (file/directory), and 'children' for directories. The output \
         is formatted with 2-space indentation. Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "excludePatterns": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["path"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let exclude = args["excludePatterns"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let valid = paths::validate_path(&path, &allowed)?;
                let tree = ops::directory_tree(&valid, &exclude)?;
                Ok(json!(serde_json::to_string_pretty(&tree)?))
            })
        }),
    );

    // ─── move_file ────────────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "move_file",
        "Move or rename files and directories. If the destination exists, the operation will \
         fail. Both source and destination must be within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "source": { "type": "string" },
                "destination": { "type": "string" }
            },
            "required": ["source", "destination"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let source = need_path(&args, "source")?;
                let destination = need_path(&args, "destination")?;
                let valid_source = paths::validate_path(&source, &allowed)?;
                let valid_destination = paths::validate_path(&destination, &allowed)?;
                ops::move_file(&valid_source, &valid_destination)?;
                Ok(json!(format!(
                    "Successfully moved {} to {}",
                    source, destination
                )))
            })
        }),
    );

    // ─── search_files ─────────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "search_files",
        "Recursively search for files and directories matching a pattern. The patterns should \
         be glob-style patterns that match paths relative to the searched directory. Use \
         '*.ext' for files in the current directory and '**/*.ext' for all subdirectories. \
         Only searches within allowed directories.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "pattern": { "type": "string" },
                "excludePatterns": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["path", "pattern"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let pattern = need_path(&args, "pattern")?;
                let exclude = args["excludePatterns"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let valid = paths::validate_path(&path, &allowed)?;
                let results = ops::search_files(&valid, &pattern, &exclude)?;
                let text = if results.is_empty() {
                    "No matches found".to_string()
                } else {
                    results
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(json!(text))
            })
        }),
    );

    // ─── get_file_info ────────────────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "get_file_info",
        "Retrieve detailed metadata about a file or directory, including size, creation time, \
         last modified time, permissions and type. Only works within allowed directories.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        Box::new(move |args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let path = need_path(&args, "path")?;
                let valid = paths::validate_path(&path, &allowed)?;
                Ok(json!(ops::get_file_info(&valid)?))
            })
        }),
    );

    // ─── list_allowed_directories ─────────────────────────────────────────
    let dirs = allowed.clone();
    server.add_tool(
        "list_allowed_directories",
        "Returns the list of directories that this server is allowed to access. \
         Subdirectories within these allowed directories are also accessible.",
        json!({ "type": "object", "properties": {} }),
        Box::new(move |_args| {
            let allowed = dirs.clone();
            Box::pin(async move {
                let dirs = allowed
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(json!(format!("Allowed directories:\n{}", dirs)))
            })
        }),
    );

    server.run().await
}
