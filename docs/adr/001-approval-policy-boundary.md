# ADR-001: The approval and policy boundary

- **Status**: Accepted (2026-10-26, openklawc v0.2 / P2)
- **Context**: Phase 2 (trust) — the policy engine

## Context

openklawc has two questions to answer before a tool call runs:

1. *Should we ask the user first?* — the autonomy gate, which already
   exists: `execute_native_tool_with_retry` in
   `crates/okc-reasoning/src/orchestrator.rs` classifies the call as
   mutating (`okc_core::is_mutation_tool`), consults the autonomy level,
   and when needed sends a `ConfirmRequest` over the agent's `confirm_tx`
   channel. The TUI answers y/n; headless runs attach no channel and the
   call is denied.

2. *Is this call permitted at all?* — the policy question. Before P2 there
   was no second layer: the mutation classification was six hardcoded tool
   names, custom tools (`sh -c`) ran ungated under `read_only`, persona
   `allowed_tools` was only prompt-deep, and there was no filesystem
   sandbox (`is_path_allowed` existed but was never called).

## Decision

**Two layers, two seams, one ordering: autonomy asks, policy denies, and
policy outranks autonomy.**

### Layer 1 — extensible mutation classification (approval path)

The existing autonomy gate stays the single approval mechanism. Its input
becomes extensible: `Agent`/`Orchestrator` carry a `mutation_extra`
`HashSet<String>` (set at boot via `with_mutation_extra`), and the gate
checks `is_mutation_tool(name) || mutation_extra.contains(name)`. Boot
populates the set from `[policy] mutating_tools` plus every
`[[tools.custom]]` entry that does not declare `mutating = false`
(custom tools are **mutating by default** — safe by default, freedom by
declaration).

*Why not a `NeedsApproval` verdict in the policy executor?* Approval
requires the confirm channel and the autonomy level, both of which live on
the agent side. Duplicating that plumbing into the executor would create
two approval flows and force `okc-policy` to depend on `okc-reasoning`
(inverted layering). Extending the classifier feeds the one existing flow
— the TUI y/n prompt — with no UI changes.

### Layer 2 — `PolicyExecutor` at the executor seam (deny path)

`crates/okc-policy` wraps the composite executor inside the timeout
guard (`Timeout(Policy(composite))`), assembled in `App::boot`. It
enforces deny-shaped rules that hold at **every** autonomy level,
including `god`:

1. **Persona restriction** — when the active persona restricts its tools,
   calls outside the set are denied at run time (not just hidden from the
   prompt).
2. **Protected paths** — globs from `[policy] protected_paths`
   (`.env`, `*.key`, `*.pem`, `secrets.*`, `~/.ssh/**`, `~/.gnupg/**`,
   `~/.config/opencode/keys/**` by default), checked in path-like
   arguments and shell command tokens.
3. **Root sandbox** — file tools may only touch paths under
   `[policy] allowed_roots` (`$CWD` plus openklawc's own data/config dirs
   by default). Relative paths are absolutized and lexically normalized,
   so `../../etc/passwd` cannot escape.

Denials return as ordinary `ToolOutcome { success: false, error }`
values — the model sees the rule it violated and can adapt; nothing
crashes, nothing is silently swallowed.

`PolicyExecutor` also publishes the previously-orphaned
`Event::FileMutation` and `Event::CommandExecuted` events on successful
mutations, which the P3 audit trail consumes.

## Consequences

- `god`/`yolo` bypass confirmation, never policy. Documented in the
  README autonomy table.
- A custom tool that is genuinely read-only opts out explicitly with
  `mutating = false`.
- Sandbox denials are configurable (`allowed_roots`), protected paths are
  list-extensible; both are deny-with-reason, machine-readable by the
  agent.
- The deny path is synchronous and cheap — verdicts return instantly,
  inside the timeout guard, and publish no events.
- Future policy rules (rate limits, per-directory tool bans) slot into
  `PolicyExecutor::check` behind config, without touching the approval
  flow.
