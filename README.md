# openklawc — Autonomous Rust Coding Agent

A Rust-native TUI coding agent that combines the feature richness of **opencode**, the clean async architecture of **openclaude**, and the autonomous agent loop of **hermes** (Nous Research).

## Architecture

openklawc is organized as a Cargo workspace with modular crates:

```
crates/
├── openklawc/      # Main binary — event loop, orchestration
├── okc-core/       # Foundational types: events, messages, tools, reasoning
├── okc-config/     # Hierarchical config loading (file, env, defaults)
├── okc-llm/        # Multi-provider LLM abstraction with streaming
│   ├── OpenAI-compatible (OpenAI, Groq, etc.)
│   ├── Anthropic Claude
│   └── Ollama (local)
├── okc-mcp/        # Model Context Protocol client (stdio + SSE)
├── okc-tool/       # Built-in tools: shell, file, git, search
├── okc-reasoning/  # Autonomy engine — plan / observe / execute / reflect
├── okc-tui/        # ratatui terminal interface
├── okc-session/    # Session persistence
├── okc-rag/        # Simple RAG / context indexing
├── okc-git/        # Git operations
└── okc-plugin/     # Plugin system (future: WASM/dynamic libs)
```

## Key Features

### Autonomy Levels
Control how much the agent can do without asking:

| Level | Behavior |
|-------|----------|
| `ask` | Confirm every tool call |
| `read_only` | Auto-execute reads, confirm mutations |
| `yolo` | Execute everything, log visibly |
| `god` | Execute everything silently (not recommended) |

### Visible Reasoning
Unlike hidden chain-of-thought, openklawc shows its reasoning inline:
- `[PLAN]` — what it intends to do
- `[OBS]` — observations from tool results
- `[EXEC]` — what it's running
- `[ERR]` — failures and recovery

### Streaming
Responses stream in real-time from any provider that supports it.

### Multi-Provider LLM
Configure multiple providers simultaneously. Switch with `/model provider/model_id`.

### MCP Support
Connect to any MCP server (Python, Node, etc.) over stdio or SSE.

### Built-in Tools
- `read_file` — Read files with line limits
- `write_file` — Create or overwrite files
- `list_dir` — Directory listings
- `search_files` — Regex search across project
- `shell` — Shell command execution
- `git` — Git operations
- `view_cwd` — Project structure summary

## Building

```bash
cd /home/bpoppa12/code/openklawc
cargo build --release
```

## Configuration

Config merges in order of priority:
1. Environment variables (`OKC_*`)
2. `.openklawc.toml` in current directory
3. `~/.config/openklawc/config.toml`
4. Built-in defaults

### Example `~/.config/openklawc/config.toml`

```toml
model = "anthropic/claude-3-5-sonnet"
small_model = "ollama/qwen3"

[provider.anthropic]
base_url = "https://api.anthropic.com/v1"
api_key = "sk-ant-..."
models = ["claude-3-5-sonnet", "claude-3-opus"]

[provider.ollama]
base_url = "http://localhost:11434"
api_key = ""
models = ["qwen3", "llama3.1"]

[mcp.web_research]
command = "python"
args = ["/path/to/web_research/mcp_server.py"]

[ui]
show_reasoning = true
stream_responses = true
vim_mode = true

[autonomy]
default_level = "read_only"
yolo_tools = ["search", "read_file", "list_dir"]
```

## Usage

```bash
# Run with default config
cargo run

# In the TUI
> How do I refactor this module?        # Chat with LLM
> @src/main.rs what does this do?       # Attach file context
/models                                  # List available models
/model anthropic/claude-3-5-sonnet      # Switch model
/autonomy yolo                           # Enable yolo mode
/rag how is auth handled?               # Search indexed context
/rag-index                              # Reindex current directory
/help                                    # Show all commands
```

## Keyboard Shortcuts

### Normal Mode (vim)
| Key | Action |
|-----|--------|
| `i` | Enter insert mode |
| `/` | Start command |
| `j` / `k` | Scroll down / up |
| `q` | Quit |

### Insert Mode (emacs)
| Key | Action |
|-----|--------|
| `Ctrl-a` / `Home` | Start of line |
| `Ctrl-e` / `End` | End of line |
| `Ctrl-k` | Kill to end |
| `Ctrl-u` | Kill to start |
| `Ctrl-w` | Kill word |
| `Enter` | Submit |
| `Esc` | Return to normal |

## Roadmap

- [ ] Tool call parsing from LLM responses (function calling)
- [ ] Full agent loop with planning/reflection
- [ ] File mutation diffing and review
- [ ] CodeMode-like sandboxed execution
- [ ] Plugin loading (WASM)
- [ ] Web interface
- [ ] Desktop app (Tauri)

## License

MIT
