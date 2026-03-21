# Ollama Is Enough: Why You Don't Need a Custom Orchestration Layer (Yet)

## The Insight

The natural instinct when designing a 3-tier model swarm is to build a sophisticated orchestration layer — a meta-controller that routes requests, manages model lifecycles, handles failover, and coordinates multi-step workflows. The previous conceptual finding (March 19) described exactly this: a tool registry, ReAct loops, an automation engine.

But for a **personal swarm of three models with one user**, all of that orchestration is premature. Ollama alone handles everything the swarm needs at this scale:

1. **Concurrent model serving**: Ollama 0.2+ supports `OLLAMA_MAX_LOADED_MODELS` (default: 3 for CPU). All three tiers can stay loaded simultaneously — the super, nano, and micro each occupying their own memory slot.

2. **OpenAI-compatible API**: Every model gets a standard API endpoint at `localhost:11434`. Claude Code, MCP servers, hooks, and any other tool that speaks OpenAI-compatible API can call any model by name.

3. **Model aliasing**: You can name models anything — `ollama pull phi4-mini` becomes your "nano," `ollama pull smollm2:135m` becomes your "micro." The names are the routing.

4. **Resource management**: Ollama handles VRAM/RAM allocation, model loading/unloading, and concurrent request queuing automatically. No Docker containers, no process management, no custom infra.

The "orchestration" for a personal swarm isn't a framework — it's a `settings.json` file and a few shell scripts. The hooks in Claude Code call the micro via `curl localhost:11434/api/chat -d '{"model":"micro",...}'`. The nano gets called the same way with `"model":"nano"`. The super gets called when the nano escalates. That's it.

**LiteLLM adds value only when you need unified routing across multiple backends** (local + cloud fallback), budget tracking, or load balancing across machines. For a single-user, single-machine swarm, it's an extra layer of complexity with no benefit. Add it later when you need cloud fallback or a second machine.

**Open WebUI adds value as the human interface** — a ChatGPT-style UI for direct conversation with any model in the swarm. It's useful for interactive knowledge management (asking the super questions about your vault, reviewing plans with the nano), but it's not required for the sidecar architecture, which operates through hooks and MCP.

## Evidence

**Ollama's concurrent model support** (from official docs and community):
- `OLLAMA_MAX_LOADED_MODELS=3` (default for CPU) — keeps 3 models hot in memory
- `OLLAMA_NUM_PARALLEL` — concurrent requests per model (default varies)
- `OLLAMA_MAX_QUEUE=512` — request queue before rejection
- Ollama 0.7 (2026) uses Go's concurrency model for efficient multi-request handling
- Ollama v0.14.0 added Anthropic Messages API compatibility — enabling Claude Code to call local models

**Memory footprint of the 3-tier swarm in Ollama**:

| Tier | Model | VRAM/RAM at Q4 | Ollama Pull Command |
|------|-------|---------------|---------------------|
| Super | Nemotron 3 Super 120B-A12B | ~60-80GB | `ollama pull nemotron-3-super` (when available) |
| Nano | Phi-4-mini (3.8B) | ~3GB | `ollama pull phi4-mini` |
| Micro | SmolLM2-135M | <1GB | `ollama pull smollm2:135m` |
| Embeddings | nomic-embed-text | ~0.3GB | `ollama pull nomic-embed-text` |

Total memory for nano + micro + embeddings: ~4.3GB. On a Mac Mini M4 16GB, this leaves 11+ GB for everything else. On the M4 Pro 64GB, there's room for the super too.

**LiteLLM's multi-model config** (from official docs):
- Maps model aliases to Ollama endpoints in `config.yaml`
- Supports routing strategies: `simple-shuffle`, `least-busy`, `usage-based`, `latency-based`
- Semantic auto-routing matches queries to models via embeddings
- Cloud fallback: route to local first, overflow to paid API when local fails
- Budget controls: per-key spending caps with auto-cutoff

LiteLLM handles 1.5k+ requests/second — far beyond what a single user needs.

**Open WebUI capabilities** (from official docs):
- 90k+ GitHub stars, most actively maintained Ollama frontend
- Multi-model conversations — switch between models mid-chat
- Built-in RAG with 9 vector database backends
- Model Builder for creating custom model presets
- Web search integration (15+ providers)
- Voice/video call features with local Whisper STT
- RBAC, LDAP, SSO support (enterprise features for later)

**The minimal orchestration stack**:

```
┌──────────────────────────────────────────┐
│           YOUR MACHINES                  │
│                                          │
│  ┌─────────────────────────────────┐     │
│  │         OLLAMA (always running) │     │
│  │                                 │     │
│  │  micro: smollm2:135m    (<1GB)  │     │
│  │  nano:  phi4-mini       (~3GB)  │     │
│  │  embed: nomic-embed-text(~0.3GB)│     │
│  │  super: nemotron-3-super(~60GB) │     │
│  │                                 │     │
│  │  API: localhost:11434           │     │
│  └──────────┬──────────────────────┘     │
│             │                            │
│  ┌──────────▼──────────────────────┐     │
│  │  CONSUMERS                      │     │
│  │                                 │     │
│  │  • Claude Code hooks (curl)     │     │
│  │  • Obsidian MCP server          │     │
│  │  • Open WebUI (optional UI)     │     │
│  │  • Smart Connections (embeddings)│    │
│  └─────────────────────────────────┘     │
└──────────────────────────────────────────┘
```

No Docker. No Kubernetes. No custom routing logic. Ollama is the entire backend. Consumers call it via HTTP. The "routing" is which model name appears in the request.

## Why This Matters

This is the anti-over-engineering insight the user explicitly asked for. The temptation is to build the orchestration layer described in the March 19 finding — tool registries, ReAct loops, automation engines. But for a swarm of 3 models serving 1 user:

- **Ollama IS the tool registry** — each model is addressable by name
- **Claude Code IS the ReAct loop** — it already reasons, acts, and observes
- **Shell scripts + hooks ARE the automation engine** — deterministic tasks don't need a framework

The path to complexity should be gradual:

```
PHASE 1 (now): Ollama + hooks + MCP
  Just Ollama serving all models. Claude Code hooks call models via curl.
  Obsidian MCP server reads/writes vault. This works immediately.

PHASE 2 (when needed): Add LiteLLM
  When you want cloud fallback (e.g., use Claude API when super is busy),
  or when you add a second machine and need cross-machine routing.

PHASE 3 (when needed): Add Open WebUI
  When you want a chat UI for direct interaction with the swarm,
  not just sidecar operations through Claude Code.

PHASE 4 (much later): Custom orchestration
  Only when the 3-model swarm grows to 5+ models with complex
  interdependencies and multi-step workflows that exceed what
  hooks + MCP can handle.
```

## Open Thread

The big unresolved question: when will Nemotron 3 Super be available as an Ollama-pullable GGUF? As of March 2026, the model was just released (March 11) and community GGUF quantizations are likely in progress but may not yet be available on the Ollama library. The older Llama-3.3-Nemotron-Super-49B-v1.5 IS available on Ollama (`mirage335/Llama-3_3-Nemotron-Super-49B-v1_5-virtuoso`). Starting with the 49B as a "super tier v0" and upgrading to the 120B MoE when available is a pragmatic path.

Also: Ollama v0.14.0's Anthropic API compatibility opens an interesting possibility — could the local swarm's nano model act as a stand-in for Claude Code on simpler tasks? If the nano handles routine operations and only escalates to the Claude API for complex reasoning, it could significantly reduce API costs. But this is scope creep for the current build — note it for later.
