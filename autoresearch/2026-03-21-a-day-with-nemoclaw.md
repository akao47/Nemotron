# A Day With Nemoclaw: What the Swarm Actually Does From Morning to Night

## The Insight

Previous findings describe Nemoclaw's architecture — the models, the machines, the vault, the hooks. But none answer the concrete question: **what does this system do for you on a random Tuesday?** Without this answer, the architecture is a solution looking for a problem.

Here's a Tuesday with Nemoclaw. Every interaction maps to a specific component from the architecture. Nothing here requires new capabilities — it's the existing components doing their jobs.

**7:30 AM — Morning Briefing (Mac Mini, always-on)**
You open Obsidian. Your daily note already has content the swarm generated overnight:
- **Session summaries**: The nano processed yesterday's 3 Claude Code sessions. "Project X: decided to use SQLite instead of PostgreSQL. Reason: simpler deployment. Changed 4 files."
- **Vault activity**: The micro detected 6 notes were modified yesterday. Tagged 3 as "image-gen", 2 as "coding", 1 as "personal."
- **Pending learning**: You dropped a ComfyUI node link two days ago. The PC processed it overnight. A personalized guide is in `vault/learning/comfyui-flux-lora-node.md`.

You didn't ask for any of this. It happened because Claude Code hooks captured sessions, the nano wrote summaries at SessionEnd, the micro classified changes when files were modified, and the 14B processed the queued link when the PC was on.

This matches the "Chief of Staff" pattern — a recently-built local-first AI assistant by a solo entrepreneur that collects Gmail, calendar, and RSS overnight, classifies everything with an LLM, and presents a ready-made briefing as an Obsidian Daily Note each morning. Nemoclaw does the same thing, but with Claude Code sessions and vault activity instead of email and calendar.

**9:00 AM — Start coding with Claude Code**
`SessionStart` hook fires. Here's what happens in <5 seconds:
1. Micro queries vault: "What project is the user in? What was the last session about?"
2. Nano compresses the answer into a 150-line briefing
3. Briefing is written to CLAUDE.md
4. Claude Code reads it automatically — it knows your last session's decisions, your project context, your preferences

You don't notice this. Claude just seems to "remember" things from yesterday. It doesn't — your sidecar told it.

**9:30 AM — Claude Code suggests something over-engineered**
Claude proposes adding a caching layer, a retry mechanism, and an abstract base class for a simple API wrapper. The `PreToolUse` hook fires before Claude writes the code:
1. Micro detects this is a Write/Edit operation on a new file
2. Nano reviews the plan against your project history: "User has a pattern of over-engineering. This feature was specced as 'simple API wrapper.' The proposed solution adds 3 abstractions."
3. Hook injects a context note: "Consider: your spec calls for a simple wrapper. Do you need the caching layer and abstract base class?"
4. Claude reads the injected context and adjusts its approach

This is the "plan review sidecar" — the nano uses your project history to catch complexity before it ships.

**11:00 AM — You find a new AI video tool on Twitter**
You copy the GitHub link and drop it into a "to-learn" note in Obsidian. What happens:
1. File change triggers the micro on Mac Mini
2. Micro classifies: "This is a URL in the to-learn folder → link-to-learning pipeline"
3. Micro scrapes the URL, searches vault for related context (your existing video gen setup, your GPU specs, your Kling workflow)
4. Writes a job file to `vault/queue/pending/learn-video-tool-xyz.json`
5. Syncthing syncs to PC. PC's file watcher picks it up.
6. 14B generates a personalized guide: "You're currently using Kling for video gen. This tool does X differently. To integrate with your workflow, you'd need to change Y. Heads up: this requires Z, which your PC can handle."
7. Guide appears in `vault/learning/video-tool-xyz.md`
8. Syncthing syncs back. Embedding index updates. The guide is searchable.

Total time: 10-20 minutes. You went back to coding. The guide appeared silently.

**2:00 PM — You ask Claude Code a question about your own setup**
"What did I decide about the image pipeline last week?" The micro intercepts this via a Claude Code hook:
1. Micro searches vault embeddings for "image pipeline decisions"
2. Finds 3 relevant notes from last week
3. Nano synthesizes: "On March 14, you decided to use Flux for img2img instead of SDXL because of better face consistency. On March 16, you added a ControlNet node to the workflow."
4. This context gets injected into Claude's window

Claude didn't remember this. The vault did. The micro found it. The nano summarized it. Claude just received the context and incorporated it into its response.

**5:30 PM — End of day**
`SessionEnd` fires on your last Claude Code session:
1. Micro classifies the day's exchanges: 12 coding decisions, 3 architectural choices, 1 bug fix
2. Nano synthesizes a session summary and writes it to the vault
3. Embedding index updates

**3:00 AM — Daily maintenance (Mac Mini, automated)**
1. Nano reads the day's 8 modified vault notes
2. Nano generates 30 instruction pairs from the new content (synthetic training data)
3. MLX runs a 15-minute LoRA fine-tune on the micro
4. New 3MB adapter swaps in. Micro's routing is now slightly better tuned to your latest patterns.

## Evidence

**The "Chief of Staff" project validates the morning briefing pattern.** Built in March 2026 by a solo entrepreneur using Claude Code + Python + SQLite, it collects Gmail, calendar, RSS feeds, and tasks overnight, classifies them with Claude Sonnet, and presents a ready-made daily briefing as an Obsidian Daily Note. The creator's motivation: "Instead of spending every morning gathering information from multiple sources, I built Chief of Staff so I can sit down and start making decisions." Nemoclaw's morning briefing is the same pattern, substituting Claude Code sessions and vault activity for email and calendar.

**The plan review sidecar has precedent in Claude Code's own hook system.** Claude Code hooks support `PreToolUse` events that fire before any tool execution. A hook can return `additionalContext` — text that gets injected into Claude's context before it proceeds. This is the mechanism for the nano's complexity warning. The latency budget is 10 seconds (hook timeout) — more than enough for the micro to route + the nano to generate a 2-sentence warning.

**The link-to-learning pipeline maps to the architecture finding's Step 1-4 workflow** (scrape → contextualize → synthesize → index). The innovation in this daily-life mapping is that the trigger is a file drop, not an API call — the user just saves a URL to a note, and the swarm takes over.

**The vault query interception pattern** is the cost-optimization insight from the micro-capability-ceiling finding: the micro routes vault-answerable questions to the local swarm instead of consuming Claude API tokens. The user doesn't decide — the micro does.

## Why This Matters

The answer to "what would I need Nemoclaw+swarm actually for?" isn't a feature list. It's this: **your AI tools become aware of each other.**

Right now, Claude Code doesn't know what you worked on yesterday. Your Obsidian vault doesn't know what Claude Code decided. Your link-to-learning tool doesn't know your existing setup. Each tool is isolated. You are the integration layer — you carry context between tools in your head, or by copy-pasting.

Nemoclaw removes you as the integration layer. The swarm's job is to:
1. **Capture** what happens in each tool (hooks, file watchers)
2. **Store** it in a shared memory (vault)
3. **Surface** it when another tool needs it (RAG + injection)
4. **Improve** its own routing over time (daily LoRA)

The daily life use cases aren't glamorous. There's no "swarm visualization" or "multi-agent collaboration." It's: you sit down, your tools already know what you did yesterday, and new things you want to learn show up as personalized guides without you asking. That's what the swarm is for.

## Open Thread

The image/video gen workflow has a specific need the current architecture doesn't fully address: **workflow versioning.** When you learn a new ComfyUI node and integrate it, your workflow changes. The vault should track not just "I learned about this node" but "my workflow went from version 3 to version 4, and here's what changed." This is a graph problem — the vault's wikilinks could represent workflow dependencies, and the nano could maintain a "current workflow" note that updates when new tools are integrated. This turns the vault into a living workflow document, not just a note collection.

Another gap: the morning briefing is passive — you read it. Could it be proactive? A scheduled message (via the Mac Mini) that sends a summary to your phone? A notification when a queued learning guide is ready? This pushes toward the "personal AI operating system" vision — not just a sidecar, but an assistant that initiates.
