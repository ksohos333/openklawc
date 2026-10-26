# OpenKLawc 5-Phase Development Plan

A roadmap to evolve OpenKLawc from its current v0.2 foundation into a robust, powerful, and highly customizable agent harness.

---

## Phase 1: Foundation & Decision Intelligence *(Current — v0.2.x)*

**Goal**: Solidify the core agent loop with fast decision validation and configurable orchestration.

### Deliverables
- [x] **Built-in tool suite** — shell, filesystem, git, search (already implemented)
- [x] **MCP client integration** — spawn and manage external tool servers
- [x] **Bundled MCP servers** — Rust ports of the opencode MCP set, auto-launched
  - `mcp-filesystem` — sandboxed read/write/edit/list/tree/search (14 tools)
  - `mcp-brave-search` — web + local search via the Brave API (2 tools)
  - `mcp-playwright` — browser automation over the Chrome DevTools Protocol (15 tools)
  - `mcp-vision` — local image understanding + audio transcription (2 tools)
  - `mcp-local-brain` — RAG memory over the local-brain HTTP service (3 tools)
  - `mcp-laya` — typed decision oracle over the TypeSafe `/v1/systemone` API (4 tools)
  - Config wins over the bundled default; `enabled = false` switches a server off
- [x] **REASON analytical framework** — structured reasoning prompts (R-E-A-S-N)
- [x] **Decision metadata extraction** — parse confidence, chosen_approach, alternatives, key_tradeoff
- [x] **Confidence-driven backtracking** — low-confidence plans trigger reconsideration
- [x] **Laya decision orchestrator** (`okc-decision` crate)
  - Crate built and unit-tested, but **unreachable at runtime** — `with_decision_layer` is never called and there is no `[decision]` config section. Wiring lands with the okc-app extraction (P1).
  - HTTP client for Ollaya daemon (`/v1/systemone`)
  - Plan validation before tool execution
  - Intent classification on user messages
  - Calibrated confidence scores in ~10ms
- [ ] **Configuration-driven decision layer**
  - `config.toml`: enable/disable decision layer per model
  - Adjustable confidence thresholds per autonomy level
  - Model selection (laya / laya:en / winnow:e4b)

### Success Criteria
- Agent can execute shell commands, read/write files, and call MCP tools
- Laya validates tool plans; blocked plans show backtrack reasoning in TUI
- All reasoning steps visible in gold-accent reasoning panel with metadata

---

## Phase 2: Advanced Agent Loop & Execution Engine *(v0.3.x)*

**Goal**: Implement production-grade execution patterns: parallel tools, retries, summarization, and configurable strategies.

### Deliverables
- [x] **Parallel tool execution** — independent tool calls run concurrently via `join_all`
- [x] **Retry with exponential backoff** — transient failures auto-retry up to `N` times with 2^n delay
- [x] **Tool choice strategies** — `auto` | `required` | `none` (config-driven)
- [x] **Tool result summarization** — long outputs (>4k chars) summarized by LLM before feedback
- [ ] **Configuration-driven behavior** — runtime honors defaults, but file values never arrive: `Agent::with_config` has no callers, so `[orchestrator]` in config is parsed then ignored. Fixed in the okc-app extraction (P1).
  - `tool_choice`, `max_iterations`, `parallel_tools`
  - `retry_count`, `retry_delay_ms`
  - `summarize_results`, `summarize_threshold`
- [ ] **Extended thinking / chain-of-thought** — native provider support for reasoning tokens (Anthropic, OpenAI o1)
- [ ] **Streaming reasoning visualization** — live reasoning tokens in the TUI
- [ ] **Computer Use patterns** — screenshot + action loop for GUI automation

### Anthropic Design Patterns (Open Source)
- **Tool use blocks** — structured input/output (already partially implemented)
- **Computer use** — vision + action loop for GUI control
- **Prompt caching** — cache system prompts across turns to reduce token cost
- **Prompt composition** — modular prompt sections that can be hot-swapped

### Success Criteria
- Agent can operate GUI applications via screenshots and predicted actions
- Provider-native reasoning tokens are visible in real-time
- Parallel tool calls reduce latency for multi-file operations

---

## Phase 3: Context, Memory & Ecosystem *(v0.4.x)*

**Goal**: Give the agent persistent memory, project awareness, and a rich tool ecosystem.

### Deliverables
- [x] **Session persistence** — SQLite store for conversation history (`okc-session`)
  - Schema: sessions, messages, reasoning_steps, full-text search index
  - Full-text search via SQLite FTS5
  - Export all sessions to JSON for backup
  - Create, list, get, delete operations
- [x] **Project context injection** — auto-detect project type and inject context
  - Detect Rust (Cargo.toml), Node (package.json), Python (pyproject.toml/requirements.txt), Go (go.mod), Java (pom.xml/build.gradle)
  - Parse dependencies for each ecosystem
  - Git awareness: branch, recent commits, uncommitted changes
  - Recently modified files
  - Injected into system prompt via `# Project Context` section
- [x] **RAG integration** (`okc-rag`)
  - In-memory TF-IDF-like word index with chunking
  - Search project files by keyword
  - Configurable depth and file extensions
- [x] **Custom tool authoring** — users can define tools in config without writing Rust
  - `[[tools.custom]]` entries in `config.toml`
  - Shell-based execution with argument substitution (`{arg_name}`)
  - Configurable timeout, working directory, and error handling
  - `CustomTools` executor integrated into `CompositeExecutor`

### Success Criteria
- Start OpenKLawc in any project directory; it understands the project automatically
- Search past conversations by keyword
- Add a new tool by editing `config.toml`, no recompile needed

---

## Phase 4: Multi-Agent & Workflow Systems *(v0.5.x)*

**Goal**: Support multiple specialized agents collaborating on complex tasks.

### Deliverables
- [x] **Chat picker (opencode-style)** — browse, resume, create and delete chats
  - Two-pane overlay: session list on the left, metadata on the right
  - Working-mode buttons (build / plan / secure) that start a new chat
  - Live search over the session list
  - Token counters and cost (when a price is configured) in the detail pane
- [x] **Agent personas / profiles**
  - Pre-defined personas: `architect`, `reviewer`, `tester`, `security_auditor`
  - Custom personas via `config.toml`
  - Persona-specific system prompts and tool access
- [ ] **Multi-agent delegation** — `delegate.rs` exists but is unreachable: no CLI flag, TUI command, or tool reaches it
  - Parent agent delegates subtasks to child agents
  - Child agents run in isolated contexts with limited tools
  - Results are aggregated back to the parent
- [ ] **Workflow engine** — `workflow.rs` parses YAML/TOML but the executor is a stub that never runs tools or LLM calls
  - YAML / TOML workflow definitions
  - Sequential steps, parallel branches, conditional logic
  - Example: `code_review.yaml` → read → analyze → comment → approve/reject
- [ ] **Agent chat mode** — multiple agents discuss a problem collaboratively
  - Round-robin turns between agents
  - Human can interject at any point
  - Consensus-based decision making with Laya validation

### Success Criteria
- Run `openklawc --persona security_auditor` for a security-focused review
- A workflow file automates a 10-step release process
- Two agents debate the best database schema; Laya scores each argument

---

## Phase 5: Specialization, Verification & Production Polish *(v0.6.x+)*

**Goal**: Transform OpenKLawc into a production-grade, trustable agent harness.

### Deliverables
- [x] **Anti-confabulation verification** (`anticonf` integration) — v1, honest scope: evidence-gathering + agent self-reporting
  - [x] Verification badge in status bar: `claims: N/M` (verified/unverified, per session)
  - [x] Agent notes claims with sources via `note_observation` (prompt section injected only when the tool is loaded)
  - [x] Legacy observation records readable again (lenient schema)
  - [ ] Automatic cross-referencing of every claim against fetched sources (verify_claim returns evidence; the verdict stays with model + human for now)
  - [ ] `[unverified]` inline prefix on uncertain claims in the scrollback
- [x] **Policy engine** — enforceable rules for agent behavior
  (`crates/okc-policy`, ADR-001)
  - Protected paths (`.env`, `*.key`, `~/.ssh/**`, …) denied at every
    autonomy level, including `god`
  - Root sandbox for file tools (`[policy] allowed_roots`)
  - Custom tools mutating by default (`mutating = false` to vouch)
  - Persona `allowed_tools` enforced at run time
  - `FileMutation` / `CommandExecuted` events published for the audit trail
  - Disallow certain tools in specific directories
  - Require confirmation for files matching `.env`, `*.key`, `secrets.*`
  - Rate-limit expensive operations (API calls, large file writes)
- [ ] **Observability & logging**
  - [x] Structured JSON logs (tracing, stderr)
  - [x] SQLite audit trail — `audit_events` + `reasoning_steps` per session
  - [x] Schema migrations with `PRAGMA user_version` and pre-migration backups
  - [x] `/audit` command in the TUI
  - [ ] Export to OpenTelemetry for external monitoring
  - [ ] Decision audit trail UI: why did the agent choose this path?
- [ ] **Integration modes**
  - **TUI** (current) — interactive terminal UI
  - **Headless** — `openklawc -p "..."` for CI/CD pipelines
  - **Daemon** — background service with HTTP API for IDE plugins
  - **Library** — embed openklawc as a crate in other Rust projects
- [ ] **Plugin system** (`okc-plugin`)
  - WASM-based plugins for custom tools and behaviors
  - Plugin marketplace
  - Hot-reload plugins without restart

### Success Criteria
- Agent claims are verified or explicitly labeled as unverified
- Policy engine prevents accidental secret leakage
- CI pipeline runs `openklawc --headless` for automated code review
- Third-party IDEs embed OpenKLawc via the daemon HTTP API

---

## Architecture Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenKLawc Harness                        │
├─────────────────────────────────────────────────────────────────┤
│  TUI │ Headless │ Daemon │ Library                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  okc-tui    │  │ okc-session │  │      okc-reasoning      │ │
│  │  (ratatui)  │  │  (SQLite)   │  │   REASON + backtrack    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ okc-decision│  │  okc-rag    │  │      okc-plugin         │ │
│  │ (Laya/     │  │ (vectors)   │  │     (WASM ext)          │ │
│  │  Ollaya)   │  │             │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     okc-llm                               │ │
│  │   OpenAI │ Anthropic │ Gemini │ Ollama │ Custom           │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     okc-tool                              │ │
│  │  Built-in (shell/fs/git)  +  MCP servers  +  Custom       │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     okc-core                              │ │
│  │   EventBus │ Types │ Traits │ Autonomy │ ReasoningStep    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decision Layer: LLM vs Laya vs Human

| Layer | Speed | Cost | Use Case |
|---|---|---|---|
| **Laya** (Ollaya) | ~10ms | Negligible | Fast validation, intent classification, safety checks |
| **LLM** (Claude/GPT) | 1-10s | Per-token | Complex reasoning, code generation, architecture decisions |
| **Human** | Variable | Time | Final approval for mutations, ambiguous situations |

The orchestrator uses Laya as a **fast filter**: if Laya says "low confidence" or "high risk", the agent pauses before invoking expensive LLM tool chains. This saves tokens and prevents hallucinated tool calls.

---

## Current Status

- [x] Phase 1: Foundation complete (.tools, MCP, REASON, Laya)
- [x] Phase 2: Advanced Agent Loop complete (parallel tools, retries, summarization, tool choice)
- [x] Phase 3: Context & Memory complete (persistence, RAG, project awareness, custom tools)
- [x] Bundled MCP servers complete (filesystem, brave-search, playwright, vision, local-brain, laya)
- [x] Phase 4 (part): personas + chat picker are real; delegation and workflows are written but unwired
- [x] Autonomy gate enforced — mutating tools need real approval in the TUI (name-based only; custom/MCP mutation tools bypass it until the policy engine, P2)
- [x] Token accounting — providers report usage, sessions track tokens and cost
- [ ] Phase 5: Production (verification, policies, observability, plugins)

## Known issues (audited 2026-10)

- Fenced JSON tool calls used to execute **twice** (fresh ids defeated the dedup guard) — fixed with (name, arguments) dedup
- Text-fallback path (Ollama / llama.cpp) double-printed the reply, raw JSON included — fixed
- `[mcp.<name>] env` values now resolve `{file:}` / `{env:}` secrets; previously the placeholder shipped literally
- Relative `[mcp.<name>] command` paths anchor to the config file's directory
- No DB migration mechanism — `PRAGMA user_version` ladder lands with the audit trail (P3)
- Local inference is migrating to llama.cpp (`llama-server` at 127.0.0.1:8080, OpenAI-compatible); Ollama provider is being removed (P5)
