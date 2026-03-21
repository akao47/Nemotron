## Research Brief

**Goal**: Should Nemoclaw use one orchestrator controlling all models, or separate Nemoclaw instances per model with a master coordinator? How does the swarm pattern apply to the personal AI architecture? And concretely, what does the user gain from Nemoclaw+swarm in their daily life?

**Source scope**: Existing autoresearch findings (especially hierarchical-personal-llm-swarm, nemoclaw-architecture-overview, claude-code-sidecar-architecture) + Web (agent orchestration patterns, multi-agent frameworks, swarm architectures)

**Iterations**: 5

**Quality bar**: Standard

**Threads to explore**:

1. **One Nemoclaw vs. many: the agent-per-model pattern**
   - Should each LLM (micro, nano, 14B) be wrapped in its own Nemoclaw agent with defined responsibilities?
   - What does a master Nemoclaw orchestrator look like that coordinates the agent-per-model instances?
   - How does this map to existing patterns: CrewAI crews, AutoGen agents, OpenAI Swarm?
   - What's the practical difference between "one process routes to models" vs "each model is an autonomous agent"?
   - When does the complexity of multi-agent orchestration pay off vs. a simple router?

2. **Swarm topology for personal use**
   - The hierarchical-personal-llm-swarm finding proposed super→nano→micro hierarchy
   - With the two-machine architecture, how does the swarm actually run?
   - What communication pattern: pub/sub, request/response, event-driven?
   - How does the swarm handle model unavailability (PC offline, model loading)?

3. **Daily life use cases: what Nemoclaw+swarm actually does for you**
   - Morning briefing: what happened overnight, what's on the agenda
   - Coding with Claude Code: persistent memory, context injection, plan review
   - Learning new tools: link-to-learning pipeline
   - Knowledge management: auto-tagging, linking, surfacing related notes
   - Image/video gen workflow: tracking new models, nodes, workflows
   - What specific user actions trigger what swarm behaviors?

4. **The automation layer: what doesn't need AI at all**
   - Which Nemoclaw operations are deterministic (no model needed)?
   - File operations, vault writes, embedding updates, sync triggers
   - The original finding's insight: "most of what agents do isn't intelligence — it's automation"
   - How to separate the intelligence layer from the automation layer in practice

5. **The personal swarm vs. cloud AI: what you gain by owning it**
   - Privacy: what specific data stays local that wouldn't with cloud AI?
   - Personalization: what can a local swarm do that Claude API + RAG can't?
   - Cost: long-term ownership economics vs. API subscription
   - The "personal AI operating system" vision from the original finding — how real is it today?

**Already covered** (avoid repetition):
- `2026-03-19-hierarchical-personal-llm-swarm.md` — Conceptual swarm architecture, UI vision, business model
- `2026-03-21-nemoclaw-architecture-overview.md` — Component diagram, data flows, costs
- `2026-03-21-daily-retraining-is-adapter-swap.md` — Training mechanics
- `2026-03-21-micro-capability-ceiling.md` — What 135M can/can't do
- `2026-03-21-link-to-learning-pipeline.md` — Tier 2 pipeline
- `2026-03-21-training-economics-free-vs-cloud.md` — Cost model
- `2026-03-21-two-machine-split-brain.md` — Two-machine architecture
- All prior sidecar, vault, orchestration findings

BUILD ON these findings. Focus on the orchestration topology, swarm patterns, and concrete personal use cases.

**Suggested starting point**: Web search for "OpenAI Swarm vs CrewAI vs AutoGen multi-agent architecture pattern 2025" to understand current swarm frameworks, then map to the Nemoclaw personal use case.

**User context**:
- Non-developer, learns fast, tends to over-engineer
- Building Nemoclaw as a personal AI system, not a product (for now)
- Already has the model/hardware architecture defined (10 prior findings)
- Needs to understand: how do the pieces actually coordinate at runtime?
- Especially cares about image/video gen tool learning workflow
