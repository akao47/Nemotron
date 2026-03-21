# Nemoclaw + Swarm Synthesis: Five Answers to Three Questions

## The Insight

Three questions launched this research session: Should each LLM have its own Nemoclaw? How does the swarm pattern apply? What is Nemoclaw+swarm actually for? Five findings answered them, and the answers converge on a single through-line: **Nemoclaw+swarm isn't a framework or a product — it's plumbing with a thin layer of intelligence on top.**

The five findings, read together, tell a story that none of them tells alone:

**1. One Nemoclaw, not many.** The agent-per-model pattern (CrewAI crews, AutoGen conversations) creates overhead that only pays off at 5-10+ agents collaborating on shared tasks. For 3 non-overlapping models serving 1 user, the handoff pattern is the right topology — the micro classifies, constructs a prompt with context, and hands off to the nano or 14B. The "orchestration" is the prompt chain, not a separate software layer. No framework needed. The micro IS the one Nemoclaw.

**2. The swarm is already event-driven.** The filesystem + Syncthing + Claude Code hooks constitute an event bus that rivals enterprise patterns (Kafka, pub/sub) at personal scale. Vault folders are topics. File watchers are subscribers. Syncthing is the transport. The file-drop pattern (write a job file, Syncthing syncs it, file watcher triggers processing) beats direct API calls for async work because it handles PC-offline gracefully, needs no authentication, and is debuggable with `ls`.

**3. A day with Nemoclaw maps every hour to a component.** Morning briefing (nano's overnight summaries + micro's auto-tagging). Coding sessions (hooks → micro classifies → nano injects context into CLAUDE.md). Over-engineering catch (PreToolUse hook → nano reviews plan). Link-to-learning (file drop → micro scrapes + retrieves context → 14B generates guide → Syncthing returns it). Daily maintenance (nano generates training data → MLX runs LoRA → micro swaps adapter). Nothing here is new capability — it's existing components connected by the plumbing.

**4. 70% of Nemoclaw is deterministic code.** Of 21 mapped operations, 12 need no model: hook dispatch, URL scraping, file watching, embedding updates, Syncthing, job queue management, templates, LoRA orchestration, Wake-on-LAN. The development priority should be inverted: build the plumbing first (weeks 1-2), add intelligence later (weeks 3-4). Each week delivers standalone value.

**5. The honest value proposition.** You gain: persistent memory that compounds ($2-7/month vs. $20-200/month cloud), verifiable privacy (data on your hardware, not Anthropic's servers), personalization that improves daily (LoRA adapters, not system prompts), and a learning pipeline that doesn't exist elsewhere. You don't gain: better reasoning (Claude still outreasons every local model), effortless setup, cutting-edge capabilities, or instant gratification.

## Evidence

The convergence across findings is the evidence. Each finding was researched independently, drawing on different sources:

- **Finding 1** drew on OpenAI Swarm, CrewAI, AutoGen architecture comparisons and the Nemotron Data Science ML Agent's tool registry pattern
- **Finding 2** drew on Confluent's event-driven multi-agent design patterns, Syncthing's capabilities, and the Unix Maildir precedent
- **Finding 3** drew on the "Chief of Staff" local-first AI assistant project (March 2026) and Claude Code's hook event system
- **Finding 4** drew on the "agentic AI vs deterministic code" industry discourse and the Nemotron Voice RAG Agent's pipeline structure
- **Finding 5** drew on 2025-2026 local vs. cloud AI cost analyses, ACLU's framing of local AI as a civil liberties issue, and capability gap assessments

Despite different sources, all five converge on the same architecture: a thin Python script (or shell scripts) that receives events from hooks and file watchers, calls the micro for classification, dispatches to the appropriate model via Ollama API, and writes results to the vault. The vault is the memory. Syncthing is the network. The filesystem is the event bus. The models provide judgment at 9 specific decision points. Everything else is code.

## Why This Matters

The synthesis resolves the original three questions definitively:

**Q: Separate Nemoclaw for each LLM?**
A: No. One Nemoclaw — the micro's routing logic, implemented as a Python script that receives events and dispatches to models. Each model is an endpoint, not an agent.

**Q: How does swarm apply?**
A: The "swarm" is the existing components communicating through the vault filesystem. No swarm framework needed. The event-driven pattern emerges from hooks (producers), file watchers (subscribers), and Syncthing (transport). The swarm is what you already have, connected.

**Q: What is Nemoclaw+swarm actually for?**
A: Making your AI tools aware of each other. Claude Code doesn't know what you worked on yesterday — the sidecar tells it. Your vault doesn't know what Claude decided — the hooks capture it. New tools you discover don't know your setup — the learning pipeline contextualizes them. You stop being the integration layer between your own tools.

The build path is clear:
1. Plumbing (week 1-2): hooks, file watchers, Syncthing, templates
2. Embeddings (week 2): nomic-embed-text, auto-update on changes
3. Micro (week 3): SmolLM2-135M for classification and routing
4. Nano + 14B (week 4+): synthesis, summaries, learning guides

Start with the boring parts. The intelligence is the last 30%.

## Open Thread

The five findings collectively surface one unresolved design question: **where does the orchestration script live?** The current architecture distributes logic across shell scripts, hook configs, file watchers, and cron jobs. Finding 4 suggested a single ~200-line Python script as the "one Nemoclaw." Finding 1 described this as "the micro's routing logic." Finding 2 described it as a file watcher that dispatches events.

These are all describing the same thing: a single long-running Python process on the Mac Mini that:
- Listens for Claude Code hook calls (HTTP endpoint or named pipe)
- Watches vault folders for file changes
- Calls the micro via Ollama API for classification
- Dispatches to nano or 14B based on classification
- Writes results to the vault
- Logs everything

This process IS Nemoclaw. Everything else is infrastructure it depends on (Ollama, Syncthing, Obsidian) or models it calls (micro, nano, 14B). The next research thread should define this process concretely — its API surface, its state management, its error handling, and its logging. Not as a framework, but as a script.
