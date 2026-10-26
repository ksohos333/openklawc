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
├── okc-mcp/            # MCP client + server framework (stdio + SSE) + bundled catalogue
├── okc-tool/           # Built-in tools: shell, file, git, search
├── okc-reasoning/      # Autonomy engine — agent loop with tool call parsing
├── okc-tui/            # ratatui terminal interface
├── okc-session/        # Session persistence
├── okc-rag/            # Simple RAG / context indexing
├── okc-git/            # Git operations
├── okc-plugin/         # Plugin system scaffold
├── mcp-web-research/   # Rust MCP: fetch URLs, web search, research loop
├── mcp-anticonf/       # Rust MCP: claim verification + observation logging
├── mcp-legal-research/ # Rust MCP: EU law + legal finding logging
├── mcp-filesystem/     # Rust MCP: sandboxed file read/write/edit/list/search
├── mcp-brave-search/   # Rust MCP: Brave web + local search
├── mcp-playwright/     # Rust MCP: browser automation over CDP
├── mcp-vision/         # Rust MCP: image understanding + audio transcription
├── mcp-local-brain/    # Rust MCP: RAG memory over local-brain
└── mcp-laya/           # Rust MCP: Laya typed decision oracle
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

All MCP servers have been rewritten in Rust and live in the workspace.

### Bundled — start automatically

These six servers launch on their own when openklawc starts. A `[mcp.<name>]`
entry in the config overrides the bundled default, and `enabled = false`
switches a server off.

| Server | Tools | Binary |
|--------|-------|--------|
| **filesystem** | `read_text_file`, `read_file`, `read_media_file`, `read_multiple_files`, `write_file`, `edit_file`, `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`, `move_file`, `search_files`, `get_file_info`, `list_allowed_directories` | `mcp-filesystem` |
| **brave-search** | `brave_web_search`, `brave_local_search` | `mcp-brave-search` |
| **playwright** | `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_fill_form`, `browser_select_option`, `browser_press_key`, `browser_evaluate`, `browser_find`, `browser_wait_for`, `browser_take_screenshot`, `browser_console_messages`, `browser_tabs`, `browser_get_config`, `browser_close` | `mcp-playwright` |
| **vision** | `vision_analyze`, `audio_transcribe` | `mcp-vision` |
| **local-brain** | `search_memory`, `recent_context`, `index_memory` | `mcp-local-brain` |
| **laya** | `laya_decide`, `laya_route`, `laya_route_batch`, `laya_status` | `mcp-laya` |

The binaries are looked up next to the `openklawc` executable, then in
`./target/release` and `./target/debug`, then on `$PATH`. Set
`OPENCLAWC_MCP_BIN_DIR` to point at a directory of MCP binaries.

Server-specific environment variables:

| Variable | Used by | Default |
|----------|---------|---------|
| `BRAVE_API_KEY` | brave-search | `~/.config/opencode/keys/brave.key` |
| `PLAYWRIGHT_CDP_ENDPOINT` | playwright | `http://127.0.0.1:9222` |
| `PLAYWRIGHT_CHROMIUM` | playwright | auto-detected in `~/.cache/ms-playwright` |
| `LOCAL_BRAIN_URL` | local-brain | `http://127.0.0.1:9876` |
| `LAYA_BASE_URL` | laya | `http://127.0.0.1:11435` |
| `LAYA_API_KEY` | laya | unset |
| `VISION_LOCAL_URL` | vision | `http://localhost:11434` |
| `VISION_MACSTUDIO_URL` | vision | `http://100.92.174.127:11434` |
| `WHISPER_SERVER_URL` | vision (audio) | `http://127.0.0.1:8080` |
| `OPENCLAWC_FS_DIRS` | filesystem | the working directory |

### Configured — start when listed in the config

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

# Override a bundled server — no `command` means "use the bundled binary".
# [mcp.filesystem]
# args = ["/path/to/openklawc", "/path/to/projects"]

# Or switch one off entirely.
# [mcp.brave-search]
# enabled = false

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
| `ask` | Every tool call needs approval |
| `read_only` | Reads run freely, mutations need approval |
| `yolo` | Execute everything, log visibly (recommended) |
| `god` | Execute everything silently |

Toggle with `/autonomy yolo` in the TUI.

**Approvals are real.** When a level requires approval the agent stops and
the TUI shows a prompt at the bottom of the screen:

```
 Approve?  y / n   write_file  {"path":"src/main.rs","content":"..."}
```

Press `y` to run the call, `n` (or `Esc`) to decline it. A declined call
never reaches the tool. In a headless run there is nobody to ask, so a call
that needs approval is denied — pass `--autonomy yolo` to allow it.

`read_file`, `list_directory`, `search_files` and friends are reads.
`write_file`, `edit_file`, `create_directory`, `move_file`, `shell` and `git`
are mutations and always need approval outside `yolo`/`god`. Custom tools
(`[[tools.custom]]`) are **mutating by default**; vouch for a read-only one
with `mutating = false`.

## Policy

Autonomy decides when to **ask**; policy decides what is **permitted at
all**. Policy outranks autonomy — a policy deny is denied in every level,
including `god`. See `docs/adr/001-approval-policy-boundary.md`.

```toml
[policy]
# Path globs that are always off-limits (any autonomy level).
# Defaults: .env, *.env, *.key, *.pem, *.pfx, secrets.*,
#           ~/.ssh/**, ~/.gnupg/**, ~/.config/opencode/keys/**
# protected_paths = ["*.tfstate", "~/vault/**"]

# Roots the file tools may touch. $CWD = working directory at boot.
# Defaults: $CWD, ~/.local/share/openklawc, ~/.config/openklawc
# allowed_roots = ["$CWD", "~/code"]

# Extra tool names treated as mutating (approval outside yolo/god) —
# e.g. MCP tools that write.
# mutating_tools = ["browser_fill_form"]
```

Denials come back to the model as ordinary tool errors naming the rule
(`policy: '~/.ssh/id_rsa' is a protected path`), so the agent can adapt.
When a persona restricts its `allowed_tools`, that restriction is enforced
at run time too — not just hidden from the prompt.

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
| `g` `g` | Jump to the bottom |
| `G` | Jump to the top |
| `c` | Open the chat picker |
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
/chats               # open the chat picker
/sessions            # browse saved sessions
/rag <query>         # search indexed context
/rag-index           # reindex current directory
/clear               # clear chat
/help                # show all commands
```

## Chat Picker

`/chats` (or `c` in normal mode) opens the opencode-style chat picker: a
list of saved chats on the left, the metadata of the highlighted chat on the
right, and a toolbar with the working-mode buttons.

| Key | Action |
|-----|--------|
| `n` | New chat in the current mode |
| `b` / `p` / `S` | New chat in build / plan / secure mode |
| `d` | Delete the highlighted chat (press `d` twice to confirm) |
| `s` or `/` | Focus the search box |
| `↑` / `↓` | Move the highlight |
| `Enter` | Resume the highlighted chat |
| `Esc` | Leave the search box, then close the picker |
| `q` | Quit |

Every turn is written to `~/.local/share/openklawc/sessions.db`, so chats
survive a restart. The detail pane shows the working mode, model, working
directory, token counters and — when a price is configured for the model —
the accumulated cost.

## Token Pricing

Cost is only shown when a price is configured for the model, so the numbers
are never guessed:

```toml
[pricing."minimax/MiniMax-M3"]
input = 1.0    # USD per 1M input tokens
output = 2.0   # USD per 1M output tokens
```

The key may be the bare model id (`MiniMax-M3`) or the qualified one
(`minimax/MiniMax-M3`); the qualified form wins.

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
