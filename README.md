# openklawc

A personal Rust TUI for an LLM agent. Talks to local MCP servers over stdio JSON-RPC. Slowly morphs toward replacing the TypeScript opencode harness it was inspired by, but starts thin and grows as it's needed.

> **Status: v0.1, Phase 1 — hello world.** This repo exists. It renders a TUI. The next phases build chat, MCP client, and per-project auto-loader. See `~/openklawc-design.md` (private notes) for the full spec.

## What it is

- A `ratatui` TUI in Rust.
- Spawns Python MCP servers (currently: `web_research`, `anticonf`, `legal_research`) as child processes.
- Speaks JSON-RPC to them. No cloud, no API keys (for the MCPs themselves — you still need an LLM API key for the agent behind them).
- One TUI, your keybindings, your theme, your auto-loader rules.

## What it is NOT

- Not a complete agent harness. v0.1 talks to existing MCPs; it doesn't yet have a full agent loop, file context, history, or tool-use routing. Those come in later phases.
- Not a fork of opencode. It's a separate binary. They can coexist; you choose which to launch.
- Not a public "platform" or product. It's a personal tool that happens to be open source because that's the right way to ship software.

## Building

You need a Rust toolchain. The project uses `uv` for the Python venvs that back the MCPs, but the Rust side is plain Cargo:

```bash
$ uv tool install rust           # if you don't have cargo/rustc yet
$ cd ~/code/openklawc
$ cargo build --release
$ ./target/release/openklawc
```

To install on `$PATH`:

```bash
$ cp target/release/openklawc ~/.local/bin/
$ openklawc
```

## Configuration

`~/.config/openklawc/config.toml`:

```toml
[mcp.web_research]
command = "/home/bpoppa12/.local/share/web_research/.venv/bin/python"
args    = ["/home/bpoppa12/.local/share/web_research/mcp_server.py"]

[mcp.anticonf]
command = "/home/bpoppa12/.local/share/anticonf/.venv/bin/python"
args    = ["/home/bpoppa12/.local/share/anticonf/mcp_server.py"]

[mcp.legal_research]
command = "/home/bpoppa12/.local/share/legal_research/.venv/bin/python"
args    = ["/home/bpoppa12/.local/share/legal_research/mcp_server.py"]

[loader.rules]
blender   = ["*.blend", "*.blend1"]
rust      = ["Cargo.toml"]
node      = ["package.json", "yarn.lock", "pnpm-lock.yaml"]
python    = ["pyproject.toml", "requirements.txt", "setup.py"]
```

A starter `config.toml` will be committed in Phase 2 with this content. For now, `cargo run` works without one (no MCPs spawned).

## Keybindings (planned — Phase 1 has only `q` to quit)

- **Normal mode (vim)**: `i` insert, `:` command, `j`/`k` scroll, `G`/`gg` jump
- **Insert mode (emacs)**: `Ctrl-a`/`Ctrl-e` line start/end, `Ctrl-k` kill to end, `Enter` submit
- **Modal (1/2/3)**: yes / no / skip for prompts
- **Arrow keys** work in both modes

See `~/openklawc-design.md` §4 for the full keymap.

## Theme

Black background, single bright gold (`#FFD700`) accent. The "bling" lives in the single highlight, not in decoration everywhere. Easy to swap if you want a different accent — `ui/theme.rs` (added in Phase 2) is the only place to change.

## Project status

- **Phase 1** (this commit): hello world. Single window, gold border, ticking counter, `q` to quit.
- **Phase 2**: chat input + scrollback with vim/emacs line editing.
- **Phase 3**: stdio JSON-RPC MCP client, wired to `web_search` first.
- **Phase 4**: per-project MCP auto-loader. `cd` into a directory with `package.json`, get the prompt to load the right MCPs.
- **Phase 5**: verify-first status badge (counts of cited vs uncited claims in the anticonf observation log).

6-week realistic plan, each phase a runnable binary. See design doc for full roadmap.

## How this was built

Honest disclosure: parts of this code (including this README and a meaningful share of the design) were generated with AI assistance. The build, debug, integration, and decisions are mine. PRs to clean up, refactor, or rewrite-from-scratch are welcome — `git log` will be honest about what's there.

## Contributing

Issues, PRs, and design feedback are all welcome. If you're submitting a PR, please run `cargo fmt` and `cargo clippy` first. The design doc is the source of truth for what's planned; if you want to build something not in the doc, open an issue first so we can sync.

## License

MIT. See `LICENSE`.
