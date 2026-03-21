# The Code Layer Beneath the Swarm: 70% of Nemoclaw Doesn't Need AI

## The Insight

Walk through every operation in the "Day With Nemoclaw" finding and ask: does this step actually need a model? The answer is surprising. Most of what makes the swarm feel intelligent is boring, deterministic code — file operations, API calls, template rendering, and cron jobs. The models provide the 30% of intelligence that makes the other 70% useful, but removing the code layer would leave the models with nothing to act on.

Here's the full operation map, split by what needs AI and what doesn't:

**Operations that need NO model (deterministic code):**

| Operation | What it does | Implementation |
|-----------|-------------|----------------|
| Hook dispatch | Catches Claude Code lifecycle events | Shell script, `curl` call |
| URL scraping | Fetches and converts web content to markdown | Crawl4ai or OneFileLLM (Python) |
| Vault file watching | Detects new/modified notes | `fswatch` or Python `watchdog` |
| Embedding index update | Re-embeds changed documents | Ollama API call with `nomic-embed-text` (embedding is deterministic for a given model) |
| Syncthing sync | Propagates files between machines | Syncthing daemon (Go binary) |
| Job queue management | Moves files between pending/processing/done folders | Shell script (`mv`) |
| Daily note template | Creates the morning briefing structure | Obsidian Templater plugin or shell script |
| LoRA training orchestration | Launches MLX training, swaps adapter file | Shell script + `cron` |
| JSONL file formatting | Structures training data as instruction pairs | Python script (string formatting) |
| Wake-on-LAN | Wakes PC for batch processing | `wakeonlan` command |
| Frontmatter updates | Adds tags/metadata to vault notes | Python or shell script (regex/YAML) |
| Log rotation | Manages session logs, training logs | Shell script + `cron` |

**Operations that need a MODEL (AI inference):**

| Operation | What it does | Which model | Why AI? |
|-----------|-------------|-------------|---------|
| Intent classification | Determines what type of request this is | Micro (135M) | Ambiguous input needs judgment |
| Vault context retrieval | Searches embeddings for relevant notes | Micro + embeddings | Semantic similarity requires understanding |
| Note tagging | Assigns project/topic tags to notes | Micro (135M) | Categories aren't keyword-matchable |
| Session summary | Synthesizes a coding session into key decisions | Nano (3.8B) | Requires comprehension + compression |
| Context briefing | Compresses vault context into 150-line CLAUDE.md | Nano (3.8B) | Requires prioritization + synthesis |
| Plan review | Checks if a proposed change is over-engineered | Nano (3.8B) | Requires reasoning about complexity |
| Instruction generation | Creates training data from vault notes | Nano (3.8B) | Requires understanding note content |
| Personalized guide | Teaches a new tool in the user's context | 14B | Requires deep comprehension + pedagogy |
| Routing decision | Decides which model handles a request | Micro (135M) | Context-dependent classification |

Count: **12 deterministic operations, 9 AI operations.** But the deterministic operations run orders of magnitude more frequently — the file watcher fires on every vault change, Syncthing runs continuously, hooks fire on every Claude Code interaction. The AI operations fire selectively, only when genuine judgment is needed.

**This means the development priority should be inverted from what intuition suggests.** The instinct is to start with the models — get the micro classifying, get the nano summarizing. But the models can't do anything useful without the code layer that feeds them. The right build order is:

1. **First: the plumbing** — hook scripts, file watchers, Syncthing, job queue folders, daily note templates
2. **Second: the embedding layer** — nomic-embed-text via Ollama, auto-update on file changes
3. **Third: the micro** — classification and routing, connected to the plumbing
4. **Fourth: the nano** — synthesis and summarization, fed by the micro's output
5. **Last: the 14B pipeline** — link-to-learning, triggered by the job queue

Each layer depends on the one below it. The 14B is useless without a job queue to feed it. The nano is useless without hooks to trigger it. The micro is useless without file watchers and embeddings to query. Build bottom-up.

## Evidence

**The "automation vs. intelligence" split is an industry consensus in 2025-2026.** The emerging pattern is "hybrid architectures" — deterministic code for structure, reliability, and safety; model inference for reasoning, adaptation, and context-awareness. A cantechit analysis captures the core tension: deterministic code gives you X in → Y out, every time. Model inference gives you contextual intelligence but introduces non-determinism. The best systems use deterministic code as the skeleton and AI as the muscles.

**The Nemotron Data Science ML Agent already implements this split.** In `chat_agent.py`, the tool registry is deterministic code — `_load_dataset`, `_train_classification`, `_show_history` are all Python functions with no model inference. The model (Nemotron Nano-9B) only decides WHICH tool to call and HOW to present the result. The actual data science operations are sklearn/cuML code.

**The Voice RAG Agent's pipeline is mostly deterministic glue.** Its 6-model chain (STT → Embeddings → Reranking → VLM → Reasoning → Safety) has deterministic steps between each model: audio preprocessing, vector store queries, prompt template construction, response formatting. The models provide intelligence at specific points; the code handles everything between.

**The original swarm finding stated this explicitly:** "Most of what agents do isn't intelligence at all — it's automation. File operations, API calls, data transforms, scheduling, routing decisions with known rules — these are deterministic tasks that don't need a model." The day-with-nemoclaw walkthrough now proves this concretely: 12 of 21 operations are pure code.

## Why This Matters

This insight protects the user from two common traps:

**Trap 1: "I need all the models working before I can use any of it."** Wrong. The code layer alone — hook scripts that save Claude Code transcripts to the vault, a file watcher that triggers embedding updates, Syncthing running between machines — provides immediate value. You get searchable session history and cross-machine file sync before any model is configured.

**Trap 2: "The models ARE the system."** Wrong. The models are specialists called on for judgment. The system is the plumbing that connects them. If you spend 3 weeks getting the micro's LoRA perfect but haven't built the file watcher that triggers it, you have a well-trained model sitting idle.

The practical implication for Nemoclaw's build sequence:

```
WEEK 1: Plumbing only (zero AI)
  • Claude Code hooks → save transcripts to vault/sessions/
  • Syncthing between Mac Mini and PC
  • File watcher on vault/queue/ folder
  • Daily note template in Obsidian
  VALUE: Searchable session history, cross-machine sync

WEEK 2: Embedding layer (AI, but simple)
  • Ollama + nomic-embed-text on Mac Mini
  • File watcher triggers re-embedding on vault changes
  • Smart Connections plugin for in-vault search
  VALUE: Semantic search across all your notes + sessions

WEEK 3: Micro intelligence
  • SmolLM2-135M via Ollama
  • Hook scripts call micro for classification
  • Micro tags sessions, routes requests
  VALUE: Auto-tagged notes, intelligent routing

WEEK 4+: Nano and 14B
  • Phi-4-mini for synthesis, summaries, plan review
  • 14B on PC for link-to-learning
  VALUE: The full "Day With Nemoclaw" experience
```

Each week delivers standalone value. If you stop after week 1, you still have something useful. This is the anti-over-engineering approach the user asked for: build the boring parts first, add intelligence gradually.

## Open Thread

The code layer raises a question about where orchestration logic lives. Right now, the "orchestration" is distributed across shell scripts, hook configs, and file watcher callbacks. As the system grows, this becomes hard to debug — "why didn't my learning guide appear?" requires tracing through 4 different scripts across 2 machines.

A single orchestration script (Python, ~200 lines) that handles all the deterministic routing could replace the distributed shell scripts. It would be the "one Nemoclaw" from the first finding — not a framework, but a single process that receives events and dispatches them. The models are still called via Ollama API. The orchestration script just decides what calls what, when, and logs everything. This is the minimum viable "framework" — a Python script with a `match` statement.
