# openklawc — Autonomous Rust Coding Agent

A Rust-native TUI coding agent that combines the feature richness of **opencode**, the clean async architecture of **openclaude**, and the autonomous agent loop of **hermes** (Nous Research).

## Architecture

```
crates/
├── openklawc/          # Main binary — event loop, orchestration
├── okc-core/           # Foundational types: events, messages, tools, reasoning
├── okc-config/         # Hierarchical config loading
├── okc-llm/            # Multi-provider LLM with streaming
│   ├── OpenAI-compatible (OpenAI, Groq, Z.AI, Kimi, DevPass, Chutes...)
│   ├── Anthropic Claude
│   └── Ollama (local)
├── okc-mcp/            # MCP client + server framework (stdio + SSE)
├── okc-tool/           # Built-in tools: shell, file, git, search
├── okc-reasoning/      # Autonomy engine — agent loop with tool call parsing
├── okc-tui/            # ratatui terminal interface
├── okc-session/        # Session persistence
├── okc-rag/            # Simple RAG / context indexing
├── okc-git/            # Git operations
├── okc-plugin/         # Plugin system scaffold
├── mcp-web-research/   # Rust MCP: fetch URLs, web search, research loop
├── mcp-anticonf/       # Rust MCP: claim verification + observation logging
└── mcp-legal-research/ # Rust MCP: EU law + legal finding logging
```

## Quick Start

```bash
cd /home/bpoppa12/code/openklawc

# Build everything (main binary + all MCP servers)
cargo build --release

# Run
cargo run --release

# Or run the binary directly
./target/release/openklawc
```

## MCP Servers (Rust-native)

All MCP servers have been rewritten in Rust and live in the workspace:

| Server | Tools | Binary |
|--------|-------|--------|
| **web_research** | `fetch_url`, `web_search`, `web_research`, `providers` | `mcp-web-research` |
| **anticonf** | `verify_claim`, `note_observation`, `list_observations` | `mcp-anticonf` |
| **legal_research** | `check_legal_status`, `fetch_eu_law`, `note_legal_finding`, `list_legal_findings` | `mcp-legal-research` |

Build them individually:
```bash
cargo build --release -p mcp-web-research
cargo build --release -p mcp-anticonf
cargo build --release -p mcp-legal-research
```

## Configuration

Config merges in order of priority:
1. Environment variables (`OKC_*`)
2. `.openklawc.toml` in current directory
3. `~/.config/openklawc/config.toml`
4. Built-in defaults

```toml
model = "ollama/qwen3"

[provider.ollama]
base_url = "http://localhost:11434/v1"
api_key = ""
models = ["qwen3"]

# Rust-native MCP servers (paths are absolute)
[mcp.web_research]
command = "/path/to/openklawc/target/release/mcp-web-research"
args = []

[mcp.anticonf]
command = "/path/to/openklawc/target/release/mcp-anticonf"
args = []

[mcp.legal_research]
command = "/path/to/openklawc/target/release/mcp-legal-research"
args = []

[autonomy]
default_level = "yolo"
```

## LLM Providers

The `okc-llm` crate supports any OpenAI-compatible API:

| Provider | Base URL Pattern | Special |
|----------|------------------|---------|
| OpenAI | `https://api.openai.com/v1` | Native function calling |
| Anthropic | `https://api.anthropic.com/v1` | Claude Messages API |
| Ollama | `http://localhost:11434` | Local inference |
| Z.AI | `https://api.z.ai/api/coding/paas/v4` | GLM models |
| MiniMax | `https://api.minimax.io/anthropic/v1` | M Plan Go |
| Kimi | `https://api.moonshot.ai/v1` | Moonshot |
| DevPass | `https://api.devpass.ai/v1` | Claude/GPT |
| Chutes | `https://llm.chutes.ai/v1` | TEE inference |
| Venice | `https://api.venice.ai/v1` | Uncensored |
| Groq | `https://api.groq.com/openai/v1` | Fast inference |

## Autonomy Levels

| Level | Behavior |
|-------|----------|
| `ask` | Confirm every tool call |
| `read_only` | Auto-execute reads, confirm mutations |
| `yolo` | Execute everything, log visibly (recommended) |
| `god` | Execute everything silently |

Toggle with `/autonomy yolo` in the TUI.

## Agent Loop

The agent loop (Hermes-inspired):
```
User Input
    ↓
PLAN  → Build system prompt with all available tools
    ↓
EXEC  → Call LLM (streaming), parse for JSON tool calls
        → Execute each tool (respecting autonomy level)
        → Feed results back as tool messages
    ↓
REFLECT → Call LLM again with tool results
    ↓
RESPOND → Return final text when no more tools needed
```

Tool calls are parsed from LLM responses as standalone JSON:
```json
{"tool": "read_file", "arguments": {"path": "src/main.rs", "limit": 50}}
```

The agent supports up to 10 tool-call iterations before forcing a response.

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

## Commands

```
/models              # list available models
/model <id>          # switch model (provider/model_id)
/autonomy <level>    # ask | read_only | yolo | god
/rag <query>         # search indexed context
/rag-index           # reindex current directory
/clear               # clear chat
/help                # show all commands
```

## File Context

Attach files to your query with `@`:
```
> @src/main.rs what does this do?
```

## Roadmap

- [x] Multi-provider LLM with streaming
- [x] Tool call parsing from LLM responses
- [x] Full agent loop (plan → execute → reflect → respond)
- [x] Built-in tools (file, shell, git, search)
- [x] MCP client (stdio + SSE)
- [x] **Rust-native MCP servers** (web_research, anticonf, legal_research)
- [ ] Native function calling (OpenAI/Anthropic tool_use)
- [ ] File mutation diffing and review
- [ ] Confirmation modal for ask/read_only modes
- [ ] CodeMode equivalent — sandboxed script execution
- [ ] Plugin loading (WASM)
- [ ] Web interface

## License

MIT
