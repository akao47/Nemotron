## Research Brief

**Goal**: Why are Tier 1 (Mac Mini M4 16GB) and Tier 2 (PC 64GB RAM, no GPU) the right hardware for a personal AI workflow combining persistent Claude Code memory, Obsidian-trained personal AI, and a link-to-learning pipeline — and how does each tier map to specific use cases?

**Source scope**: Web (Ollama docs, HuggingFace model cards, RunPod pricing, MLX/llama.cpp benchmarks, Obsidian plugin docs) + existing autoresearch findings

**Iterations**: 5

**Quality bar**: Standard

**Threads to explore**:

1. **Tier 1 training reality: daily Obsidian retraining on M4 16GB**
   - What does "train daily on Obsidian vault" actually look like at 50M-135M params?
   - LoRA fine-tune vs full retrain vs RAG-only (which approach fits daily cadence?)
   - Practical training loop: vault markdown → training data format → train → deploy
   - Time/resource cost per daily cycle on M4 16GB with MPS backend
   - Why 50M-135M is the ceiling for daily local retraining (not 1B, not 350M)

2. **Tier 1 use case mapping: persistent memory + personal AI**
   - Use case 1: Persistent memory for Claude Code sessions (micro as sidecar)
   - Use case 2: Personal AI trained on user's knowledge and preferences
   - Why both fit on M4 16GB simultaneously (memory budget breakdown)
   - What "trained on my information" means at micro scale — capabilities and limits
   - When does the micro say "I don't know, ask the nano"?

3. **Tier 2 use case: the link-to-learning pipeline**
   - User sends a link (GitHub repo, new tool, update) → AI teaches them how to use it
   - Why this requires a larger model (7B-14B) that Tier 1 can't handle well
   - The pipeline: URL → scrape/fetch → comprehend → map to user's workflow → explain
   - Why a 135M micro can't do "teach me how to use this new tool in my setup"
   - Why 14B-27B running on 64GB RAM CPU handles this (speed tradeoff is acceptable)
   - Image/video gen tool learning: what model capabilities are needed

4. **Tier 2 training economics: RunPod for what you can't train locally**
   - LoRA fine-tuning 7B-14B on RunPod: cost per session, how often needed
   - Contrast with Tier 1: daily local retrain (free) vs periodic cloud fine-tune ($7-40)
   - When do you need to fine-tune the Tier 2 model vs just using RAG?
   - The hybrid: Tier 1 micro retrained daily (free), Tier 2 nano fine-tuned monthly (RunPod)

5. **The two-machine architecture: why not just one bigger machine?**
   - Mac Mini as always-on sidecar (low power, silent, persistent)
   - PC as on-demand powerhouse (higher capability, used when needed)
   - How they communicate: shared Obsidian vault, Ollama API, local network
   - The upgrade path: what changes when you eventually add a GPU to the PC

**Already covered** (avoid repetition):
- `2026-03-19-hierarchical-personal-llm-swarm.md` — Conceptual swarm architecture
- `2026-03-21-hardware-matrix-moe-changes-everything.md` — MoE hardware implications
- `2026-03-21-model-selection-the-135m-pivot.md` — SmolLM2-135M as micro, Phi-4-mini as nano
- `2026-03-21-obsidian-vault-as-swarm-memory.md` — Three-layer Obsidian integration
- `2026-03-21-claude-code-sidecar-architecture.md` — Hook-based memory pipeline
- `2026-03-21-orchestration-ollama-is-enough.md` — Ollama as serving layer

Do NOT re-derive model selection, vault integration layers, or sidecar architecture. BUILD ON those findings with hardware-specific practical details.

**Suggested starting point**: Web search for "LoRA fine-tuning SmolLM2-135M on custom data" and "Apple M4 MPS training benchmarks" to anchor the Tier 1 daily retraining thread with real numbers.

**User context**:
- Non-developer, learns fast, tends to over-engineer
- Already has Mac Mini M4 16GB
- Planning to get PC with 64GB RAM (no GPU initially)
- Uses Claude Code as primary dev tool
- Uses Obsidian vault as knowledge base
- Especially interested in image/video generation tools
- Wants to drop links to new tools and have AI teach them how to use it their way
- Super tier (Nemotron Ultra 253B) is a future dream — not part of this research session
