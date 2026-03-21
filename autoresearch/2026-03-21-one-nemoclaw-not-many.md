# One Nemoclaw, Not Many: Why the Handoff Pattern Beats Agent-Per-Model

## The Insight

The question "should each model have its own Nemoclaw?" maps to a well-studied architecture choice in multi-agent systems. There are three topologies on the table:

1. **Router pattern** — one central process dispatches requests to models based on simple rules. Models are dumb endpoints. The router decides everything. This is what the "Ollama is enough" finding already describes: `curl` calls to named models, routing logic in shell scripts.

2. **Agent-per-model pattern** — each model gets its own autonomous agent wrapper (its own Nemoclaw) with state, memory, and decision-making ability. A master Nemoclaw coordinates the agents. This is what CrewAI does with "crews" — each agent has a role, tools, and memory. It's also what AutoGen does with conversing agents.

3. **Handoff pattern** — agents transfer control to each other through explicit "transfer_to_X" functions. Each agent handles its piece and passes the conversation along. No master orchestrator — just a chain of handoffs. This is what OpenAI Swarm pioneered and the Agents SDK formalized.

For Nemoclaw's 3-model personal swarm, **the handoff pattern is the right topology — but you don't need a framework to implement it.** Here's why:

The agent-per-model pattern (option 2) creates overhead that only pays off at scale. CrewAI-style crews with role definitions, task objects, process flows, and memory layers make sense when you have 5-10 agents working on a complex task with interdependencies — a research crew with a planner, researcher, writer, and editor. But Nemoclaw has three models with non-overlapping capabilities: the micro classifies, the nano thinks, the 14B teaches. They don't collaborate on the same task. They hand off to each other.

The router pattern (option 1) is what's already been described in prior findings, and it works for the initial implementation. But it caps out when the micro needs to make contextual routing decisions — "this question looks like it's about ComfyUI, but the user's vault says they're working on a Kling video project today, so route to the video context instead." That's not a simple rule. That's a handoff with context.

The handoff pattern (option 3) is the sweet spot. Each model is an agent with a defined capability boundary. The micro doesn't just route — it makes a handoff decision and passes along the relevant context. "I classified this as a learning request about ComfyUI. Here's the user's existing vault context about their ComfyUI setup. Handing off to the 14B." The 14B doesn't need to know about the micro's classification logic. It just receives a well-prepared prompt.

**The critical implementation detail: the handoff IS just a well-constructed prompt.** You don't need a framework for this. The micro's output becomes the nano's input. The nano's synthesis becomes the 14B's context. Each model transforms and passes the baton. The "orchestration" is the prompt chain, not a separate software layer.

## Evidence

**OpenAI Swarm's core pattern is surprisingly minimal.** An agent is just: instructions (system prompt) + tools (functions it can call) + handoffs (functions that return the next agent). The entire framework is ~300 lines of Python. OpenAI explicitly called it "educational" — the pattern is the product, not the code. The Agents SDK that replaced it adds production concerns (tracing, guardrails) but keeps the same two primitives: agents and handoffs.

**The Nemotron Data Science ML Agent (`use-case-examples/Data Science ML Agent/src/chat_agent.py`) already implements the router pattern** — a single `ChatAgent` class with a `self.tools` dictionary mapping operation names to handler functions. One agent, one model (Nemotron Nano-9B), many tools. This is the simplest topology — and it works because the Data Science Agent only needs one model's capabilities.

**The Voice RAG Agent implements a pipeline, not a swarm.** Its 6-model chain (STT → Embeddings → Reranking → VLM → Reasoning → Safety) is sequential — each model processes and passes to the next. No model decides which model runs next. No handoffs. Just a fixed pipeline. This works because the flow is deterministic — every voice query goes through the same stages.

**Nemoclaw needs something between these two.** Unlike the Data Science Agent, Nemoclaw has multiple models with different capabilities. Unlike the Voice RAG pipeline, Nemoclaw's routing is contextual — a query might go to the nano OR the 14B depending on what it is. The handoff pattern captures exactly this: the micro evaluates context, decides the next agent, and passes control with relevant context attached.

**CrewAI and AutoGen would be over-engineering.** CrewAI's crew metaphor requires defining explicit roles, tasks, process flows, and layered memory (ChromaDB, SQLite, vector embeddings). AutoGen's conversation model requires agents that chat with each other in multi-turn exchanges. For a single user with three non-overlapping models, this creates more architecture to maintain than value delivered. The frameworks are designed for 5-10+ agents collaborating on shared tasks — not 3 models in a clear hierarchy.

**The "one Nemoclaw" implementation is just the micro's routing logic expanded:**

```
USER INPUT
    │
    ▼
MICRO (SmolLM2-135M)
    │
    ├── classify intent
    ├── search vault for context
    ├── decide: who handles this?
    │
    ├──[retrieval] → return vault results directly
    ├──[explain/synthesize] → handoff to NANO with vault context
    ├──[learn new tool] → handoff to 14B with scraped content + vault context
    └──[complex reasoning] → handoff to Claude Code (via hooks)
```

The "one Nemoclaw" is the micro. It's the bouncer that already exists in the architecture. The handoff is the prompt it constructs for the next model. No framework, no crew definitions, no agent-to-agent chat protocols.

## Why This Matters

This resolves a common trap in AI system design: **confusing model coordination with agent orchestration.** When you have models that don't overlap in capability, you don't need agents that negotiate. You need a router that knows which specialist to call. The handoff pattern gives the router just enough intelligence to pass context — without the overhead of autonomous agents managing their own state and memory.

For a personal system, the rule is: **complexity in the models, simplicity in the glue.** The micro's LoRA adapter gets retrained daily — that's where the personalization lives. The orchestration between models should be boring shell scripts and `curl` calls with well-constructed prompts. If you find yourself building a CrewAI crew or an AutoGen conversation flow for three models and one user, you're writing framework code instead of using your system.

The practical "one Nemoclaw" implementation is:
1. A Python script (or shell script) that receives user input
2. Calls the micro via Ollama API to classify + retrieve context
3. Based on classification, constructs a prompt for the appropriate next model
4. Calls that model via Ollama API
5. Returns the result

That's the entire orchestration. No state machine. No agent framework. No coordination protocol. Just a micro that decides and a prompt that carries context.

## Open Thread

The handoff pattern breaks down when you need **multi-model collaboration** — when the nano's output needs to be reviewed by the micro before going to the user, or when the 14B's guide needs to be validated against the vault by the nano. These are loops, not handoffs. The question becomes: at what point does Nemoclaw's workflow require loops? The sidecar's "capture → process → inject" pipeline already has one (micro classifies → nano synthesizes → micro retrieves next session). If more loops emerge, the handoff pattern may need to evolve into an event-driven pattern where models subscribe to events rather than receiving handoffs. But that's a future problem — start with handoffs, add loops when specific workflows demand them.
