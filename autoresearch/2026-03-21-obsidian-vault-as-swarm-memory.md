# The Vault Is the Memory Bus: Three Layers of Obsidian Integration for a Local Swarm

## The Insight

Connecting a local model swarm to an Obsidian vault isn't a single integration problem — it's three distinct layers, each solving a different memory function. Most guides treat it as "hook up RAG and you're done," but a persistent sidecar needs all three:

**Layer 1: Semantic Search (read-only memory recall)**
The vault is already full of notes. The swarm needs to find relevant ones based on meaning, not just keywords. This is the classic RAG layer — embed the vault, search by similarity. Smart Connections plugin already does this out-of-the-box with its built-in BGE-micro-v2 embedding model (no setup required, ~786K downloads). For the swarm, a dedicated embedding model like nomic-embed-text (~137M params, runs via Ollama) provides better quality and can be shared across all three tiers.

**Layer 2: Structured Read/Write (active memory operations)**
The sidecar doesn't just read — it needs to write session summaries, update project status notes, tag notes with new metadata, and create new notes when it learns something relevant. This is where MCP servers come in. The `cyanheads/obsidian-mcp-server` provides 54+ tools for reading, writing, searching, and managing notes through the Obsidian Local REST API. The sidecar can:
- Read notes to build context before a Claude Code session
- Write session summaries after a Claude Code session ends
- Update frontmatter tags (project status, relevance scores, last-reviewed dates)
- Create new notes when it ingests new AI developments from the user's feeds

**Layer 3: Graph-Aware Context (relational memory)**
Obsidian's `[[wikilinks]]` create an explicit knowledge graph. Standard vector search ignores this structure — it treats every note as an isolated document. Graph RAG approaches (like ObsidianRAG's GraphRAG or Neural Composer's LightRAG integration) follow wikilinks to expand context, surfacing notes that are semantically distant but structurally connected. This is critical for a knowledge management sidecar: when the user asks "what relates to Project X," the answer should follow the link structure, not just embedding similarity.

The insight is that these three layers map directly to the three model tiers:

- **Micro (SmolLM2-135M)**: Handles Layer 1 operations — classification, routing, simple retrieval queries. Determines *which* notes are relevant and *what type* of operation is needed.
- **Nano (Phi-4-mini)**: Handles Layer 2 operations — reading note content, synthesizing context, writing structured summaries, updating frontmatter. Needs language understanding but not deep reasoning.
- **Super (Nemotron 3 Super)**: Handles Layer 3 operations — graph traversal reasoning, connecting disparate knowledge threads, answering complex "how does X relate to Y" questions that require multi-hop reasoning across the vault.

## Evidence

**MCP Server ecosystem for Obsidian** (from official repos):

| Project | Approach | Tools | Requires |
|---------|----------|-------|----------|
| cyanheads/obsidian-mcp-server | REST API bridge | 54+ (read, write, search, tags, frontmatter) | Obsidian Local REST API plugin |
| Storks/obsidian-mcp | Obsidian CLI bridge | 54 tools | Obsidian desktop app running |
| obsidian-notes-rag | MCP + vector search | Semantic search, embeddings | sqlite-vec, Ollama/LM Studio |

The cyanheads MCP server includes an intelligent in-memory vault cache that provides fallback for global search if the live API fails — important for an always-on sidecar.

**RAG/Vector Search plugins**:

| Plugin | Embedding Model | Vector Store | Graph-Aware? |
|--------|----------------|-------------|-------------|
| Smart Connections | BGE-micro-v2 (built-in) | Local (.smart-env/) | No |
| ObsidianRAG | Via Ollama (configurable) | Hybrid (Vector + BM25) | Yes (wikilinks) |
| Neural Composer | Via LightRAG | Graph-based | Yes (GraphRAG) |
| Obsidian Notes RAG | Ollama/OpenAI/LM Studio | sqlite-vec | No |

Smart Connections has 786K+ downloads and works out-of-box with zero configuration. ObsidianRAG adds hybrid search and wikilink-aware context expansion. Neural Composer uses LightRAG for graph-based RAG.

**Embedding model options for local deployment**:

| Model | Params | Purpose |
|-------|--------|---------|
| BGE-micro-v2 | ~33M | Smart Connections default, zero-config |
| nomic-embed-text | ~137M | Higher quality, runs via Ollama |
| BAAI/bge-m3 | ~568M | Multilingual, higher quality |

## Why This Matters

The three-layer model means the Obsidian vault isn't just a data source — it's the **shared memory bus** for the entire swarm. Every tier reads from it, some tiers write to it, and the graph structure provides the relational context that flat vector search misses.

The practical wiring looks like:

```
┌─────────────────────────────────────────┐
│              OBSIDIAN VAULT             │
│  (markdown files + wikilinks + tags)    │
├─────────────────────────────────────────┤
│                                         │
│  Layer 1: Embeddings (nomic-embed-text) │
│  → Semantic search across all notes     │
│  → Triggered by micro for routing       │
│                                         │
│  Layer 2: MCP Server (cyanheads)        │
│  → Read/write/search/tag operations     │
│  → Used by nano for context building    │
│  → Used by nano for session summaries   │
│                                         │
│  Layer 3: Graph RAG (ObsidianRAG)       │
│  → Wikilink-aware context expansion     │
│  → Used by super for deep reasoning     │
│                                         │
└─────────────────────────────────────────┘
```

The user already has the Obsidian vault. The integration stack is:
1. Install Obsidian Local REST API plugin (one click)
2. Run cyanheads MCP server (npm install, configure)
3. Run Ollama with nomic-embed-text for embeddings
4. Install Smart Connections or ObsidianRAG for in-vault search UI

All of this runs locally. No cloud, no API keys, no data leaving the machine.

## Open Thread

The write path creates a consistency challenge. If the nano writes a session summary and the micro's embedding index isn't updated, subsequent searches will miss the new note. The embedding index needs to be incrementally updated when new notes are created — Smart Connections does this automatically (it watches for file changes), but a custom embedding pipeline would need a file watcher or webhook trigger.

Also: the vault's wikilink graph is user-curated and potentially inconsistent. The sidecar could help maintain graph hygiene — suggesting links between related notes, flagging orphaned notes, identifying clusters that should be connected. This is a natural role for the nano tier: enough language understanding to assess relatedness, but not requiring the super's reasoning capability.
