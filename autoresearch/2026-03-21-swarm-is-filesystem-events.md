# The Swarm Is Already Event-Driven: Your Filesystem Is the Message Bus

## The Insight

Enterprise multi-agent systems use Kafka, RabbitMQ, or AWS EventBridge as their event backbone — a central message broker where agents publish what they learn and subscribe to what they need. Confluent's 2025 design patterns for multi-agent systems describe "topic-based workflows" where events flow through semantic channels and new agents can join by subscribing to relevant topics. This is the right architecture for 50 agents serving 10,000 users.

For Nemoclaw's 3-model personal swarm on 2 machines, the event backbone already exists: **the filesystem + Syncthing + Claude Code hooks.** You don't need to build an event system because you already have one. You just haven't named it yet.

Here's what's already event-driven in the architecture:

**Filesystem events (always-on, Mac Mini):**
- A new file appears in the vault → the embedding index updates (Smart Connections watches for file changes)
- A file is modified → Syncthing propagates the change to the PC within seconds
- A new session summary lands in `vault/sessions/` → the micro's next LoRA training batch includes it

**Claude Code hook events (on-demand, during coding):**
- `SessionStart` → micro retrieves context, nano generates briefing, writes to CLAUDE.md
- `Stop` (every response) → micro classifies the exchange, tags for memory
- `PreCompact` → nano saves important context before compression
- `SessionEnd` → nano writes final session summary to vault

**Syncthing events (cross-machine, continuous):**
- PC writes a learning guide to the vault → Syncthing syncs to Mac Mini → embedding index auto-updates → guide is searchable from the sidecar
- Mac Mini's micro classifies a link-to-learning request → writes scraped content + context to a "pending" folder → Syncthing syncs to PC → a file watcher on the PC triggers the 14B

**The key realization: the communication pattern between the two machines isn't an API call — it's a file drop.** The Mac Mini doesn't need to call the PC's Ollama API directly (though it can). The simpler, more resilient pattern is:

1. Mac Mini writes a job file to `vault/queue/pending/learn-comfyui-node-xyz.json`
2. Syncthing syncs the file to the PC in 1-5 seconds
3. A file watcher on the PC picks it up, calls the local 14B via Ollama
4. The 14B's output is written to `vault/learning/comfyui-node-xyz.md`
5. Syncthing syncs the output back to the Mac Mini
6. The Mac Mini's embedding index auto-updates

No API endpoint to expose. No firewall rules. No `OLLAMA_HOST=0.0.0.0`. No authentication concerns. The vault IS the message queue. Syncthing IS the transport. The filesystem IS the event bus.

This is the "topic-based workflow" pattern from enterprise event-driven architecture, implemented with zero infrastructure: the folder path IS the topic. `vault/queue/pending/` is the "work requests" topic. `vault/learning/` is the "completed guides" topic. `vault/sessions/` is the "session memories" topic. Any component can publish to any topic by writing a file. Any component can subscribe by watching a folder.

## Evidence

**Syncthing as an event transport is well-documented.** Syncthing provides real-time, bidirectional file sync over LAN with <1 second detection latency for changes on most platforms. It supports folder-level granularity — you can sync only the `vault/queue/` folder at high frequency while syncing the rest of the vault at normal intervals. Conflict resolution is deterministic (newest wins, or manual via the Syncthing UI).

**Filesystem watchers are native on both platforms:**
- macOS: `fswatch` or Python's `watchdog` library — watches for new files in a directory and triggers a callback
- Linux/Windows: `inotifywait` (Linux) or Python's `watchdog` — same pattern
- Ollama itself doesn't need a watcher — the watcher calls Ollama when a job file appears

**The "file as message" pattern has precedent.** Unix mail systems used Maildir — each email was a file, folders were queues, and processing was file operations. Printing systems use spool directories. CI/CD systems use artifact directories. The pattern is old, reliable, and debuggable — you can `ls` your message queue.

**Direct API calls (the alternative) require infrastructure the file pattern avoids:**

| Concern | File-drop pattern | Direct API pattern |
|---------|------------------|--------------------|
| PC offline | Job file waits in queue. PC processes when it wakes. | Request fails. Need retry logic + queue anyway. |
| Authentication | None needed — filesystem permissions | Ollama has no auth. Need reverse proxy or firewall rules. |
| Debugging | `ls vault/queue/pending/` — see what's waiting | Check API logs, trace request/response |
| State persistence | Job files are the state. Survives reboot. | Need separate state store. |
| PC address changes | Syncthing handles discovery | Need to update IP in config |

The file-drop pattern is strictly superior for an asynchronous, batch-oriented pipeline like link-to-learning. The PC doesn't need to be reachable by IP. It doesn't need to be on. It just needs to sync when it's up.

**When direct API IS better:** Real-time queries that need <1 second round-trip. If the micro needs to ask the 14B a follow-up question during a Claude Code session (not the current design, but a possible future need), the file-drop pattern's 1-5 second Syncthing latency is too slow. For this, direct Ollama API over LAN (`http://pc-ip:11434`) is the right choice. The architecture should support both: file-drop for batch/async work, direct API for rare real-time needs.

## Why This Matters

The "swarm" isn't a thing you build on top of the existing components. **The swarm IS the existing components, communicating through the vault.** When you name this pattern, the architecture becomes clearer:

- The Obsidian vault isn't just storage — it's the **event bus and message queue**
- Syncthing isn't just sync — it's the **transport layer**
- Folder paths aren't just organization — they're **topics/channels**
- File watchers aren't just triggers — they're **event subscribers**
- Claude Code hooks aren't just callbacks — they're **event producers**

This means "building the swarm" is really just:
1. Define the folder structure (topics)
2. Write the file watchers (subscribers)
3. Configure the hook scripts (producers)
4. Let Syncthing handle the rest (transport)

No Kafka. No message broker. No event bus library. No custom protocol. The filesystem does it all — and it's debuggable with `ls`, `cat`, and `tail -f`.

The enterprise event-driven architecture literature is right about the pattern (pub/sub beats request/response for multi-agent coordination). It's just wrong about the implementation for a personal system. You don't need Kafka for 3 subscribers and 1 user. You need folders and file watchers.

## Open Thread

The file-drop pattern introduces a question about job format. What goes in a job file? A JSON blob with the request type, input content, vault context, and target model? Or a markdown file that the 14B can read directly? The simpler option is markdown — the 14B receives a file that looks like a prompt, generates a response, and writes it as another markdown file. The micro on the Mac Mini handles the JSON complexity (vault context assembly, classification metadata) and outputs human-readable markdown that any model can consume. This keeps the inter-machine protocol dead simple: markdown in, markdown out.

Another thread: job ordering and deduplication. If the user drops three links in quick succession, three job files appear. Do they run sequentially? In parallel (if the PC has enough RAM for concurrent 14B calls)? Does the system deduplicate if two links point to the same content? For a single user dropping 2-3 links a week, this is a non-problem. But it's worth noting that the file-drop pattern naturally supports sequential processing (process files in creation-time order) without any additional infrastructure.
