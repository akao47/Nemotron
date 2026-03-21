# The Sidecar Pattern: How a Local Swarm Becomes Claude Code's Long-Term Memory

## The Insight

Claude Code already has the attachment points for a persistent memory sidecar — hooks (SessionStart, Stop, PreCompact, SessionEnd), MCP server support, and CLAUDE.md auto-injection. What's missing isn't infrastructure; it's **an intelligent process that decides what to remember, what to forget, and what to surface at the right moment**. This is exactly the job the local swarm was designed for.

The existing community solutions (mcp-memory-keeper, memory-mcp, mcp-memory-service) all use the same basic pattern: hook into Claude Code's lifecycle events, extract memories from the conversation transcript, store them, and inject them on the next session start. But they all use cloud LLMs (typically Haiku) for the extraction step, and they all treat memory as a flat store — facts go in, facts come out.

A local swarm sidecar does something fundamentally different. It creates a **three-stage memory pipeline** that runs entirely locally:

**Stage 1 — Capture (hooks → micro)**
Claude Code hooks fire on `Stop` (every response), `PreCompact` (before context compression), and `SessionEnd`. The hook sends the recent transcript to the micro (SmolLM2-135M), which classifies each exchange: is this a decision? A preference? A bug fix? A project update? An architectural choice? The micro doesn't understand the content deeply — it tags it for downstream processing.

**Stage 2 — Process (micro output → nano)**
The nano (Phi-4-mini) receives the tagged exchanges and does the actual synthesis: extracts the key information, deduplicates against existing memories, assesses relevance and priority, and writes structured notes to the Obsidian vault. Because the nano understands code and can reason about project context, it produces higher-quality summaries than simple LLM extraction.

**Stage 3 — Inject (vault → CLAUDE.md)**
On `SessionStart`, the micro queries the Obsidian vault for context relevant to the current project/directory. The nano synthesizes the top-k results into a compact briefing (~150 lines). This briefing is written to CLAUDE.md (or injected via the hook's `additionalContext` field in PreToolUse). Claude Code reads it automatically — no MCP call needed, no token overhead.

The key difference from existing memory MCP servers: **the processing happens in the swarm, not in Claude Code's context window**. Memory extraction, deduplication, prioritization, and synthesis all happen outside Claude's token budget. Claude Code only sees the final, compressed briefing — the distilled result of the sidecar's work.

## Evidence

**Claude Code hook events available for sidecar integration**:

| Event | When | Sidecar Use |
|-------|------|-------------|
| SessionStart | Session begins | Inject context briefing via CLAUDE.md or additionalContext |
| PreToolUse | Before any tool call | Add relevant memories as context for specific operations |
| PostToolUse | After tool execution | Capture results (e.g., test outcomes, file changes) |
| Stop | After each Claude response | Extract memories from latest exchange |
| PreCompact | Before context compression | Save important context before it's compressed away |
| SessionEnd | Session terminates | Final memory consolidation and vault sync |

Hook handlers can be `command` (shell script), `prompt` (semantic evaluation), or `agent` (deep analysis with tool access). The sidecar would use `command` handlers that call the local swarm's API.

**Existing memory MCP server patterns** (from community projects):

The `memory-mcp` by yuvalsuede uses a two-tier system:
- Tier 1: CLAUDE.md (~150 lines) — compact briefing, auto-generated
- Tier 2: .memory/state.json (unlimited) — full memory store

80% of sessions need only Tier 1. This validates the "inject a briefing, not the full memory" approach.

The `mcp-memory-service` by doobidoo provides session-start.js and session-end.js hooks that load/save context automatically. It uses sqlite-vec for semantic search over stored memories.

**CLAUDE.md injection mechanics**:
- Claude Code reads CLAUDE.md from the project root and ~/.claude/ on every session start
- This is automatic — no configuration needed
- Content in CLAUDE.md becomes part of Claude's system context
- The sidecar can write to CLAUDE.md between sessions to update the briefing

**PreToolUse additionalContext** (v2.1.9+):
- Hooks can return `additionalContext` to inject information before tool execution
- This enables targeted memory injection — e.g., when Claude is about to read a file, the sidecar can inject "last time you worked on this file, you decided X because Y"
- This is more precise than bulk CLAUDE.md injection

## Why This Matters

The sidecar architecture solves three problems that Claude Code's built-in auto-memory doesn't:

1. **Depth of processing**: Auto-memory saves surface-level observations. The sidecar's nano tier can reason about project architecture, track multi-session decision threads, and maintain a coherent project narrative.

2. **Cross-project memory**: Auto-memory is per-project. The sidecar's Obsidian vault spans all projects — when you learn something in Project A that's relevant to Project B, the sidecar can surface it. This is the "ingests new AI developments and maps them to personal projects" use case.

3. **Anti-over-engineering check**: The user explicitly wants the sidecar to review spec/plans before implementation. On `PreToolUse` for Write/Edit operations, the sidecar can inject a reminder: "Your spec for this feature calls for X, but based on your history of over-engineering, consider whether you actually need Y." This requires the nano's reasoning capability — the micro can't assess engineering complexity.

The practical implementation sequence:

```
WEEK 1: Basic memory loop
  SessionEnd hook → save transcript summary to Obsidian
  SessionStart hook → inject last session's summary into CLAUDE.md

WEEK 2: Intelligent extraction
  Replace "save transcript" with micro classification
  Add nano synthesis for higher-quality memories
  Implement deduplication against existing vault notes

WEEK 3: Proactive context injection
  PreToolUse hooks for file operations
  Inject relevant memories contextually, not just at startup
  Add cross-project memory surfacing

WEEK 4: Plan review sidecar
  PreToolUse hook for Write operations on spec/plan files
  Nano reviews the plan against project history
  Injects complexity warnings or simplification suggestions
```

## Open Thread

The latency question matters. Claude Code hooks have a 10-minute timeout, but users expect sub-second responsiveness. The micro (SmolLM2-135M at ~724MB) processes near-instantly on CPU. The nano (Phi-4-mini at ~3GB) takes 1-5 seconds for short synthesis tasks. For `SessionStart` injection, this latency is fine — the user waits a few seconds anyway. For `PreToolUse` injection, it might feel sluggish if the nano runs on every tool call. The solution may be to use the micro for routing (instant) and only invoke the nano for high-value interventions (plan reviews, architecture decisions).

Another consideration: the hook's `additionalContext` field adds tokens to Claude's context window. Too much injected context wastes the user's token budget. The nano needs to be trained (or prompted) to produce extremely compact summaries — 2-3 sentences per injection, not paragraphs.
