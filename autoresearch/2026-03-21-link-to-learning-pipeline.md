# The Link-to-Learning Pipeline: Why "Teach Me This Tool" Needs 14B and a Different Machine

## The Insight

The "send a link and have AI teach me how to use it" use case has fundamentally different requirements from the persistent memory sidecar — and those differences explain why it needs Tier 2 (PC 64GB RAM) instead of Tier 1 (Mac Mini M4 16GB).

The sidecar is **reactive and fast**: it classifies, routes, and retrieves in the background of a coding session. Latency matters — sub-second for the micro, 1-5 seconds for the nano. The link-to-learning pipeline is **proactive and slow**: you send a link, go make coffee, come back to a personalized tutorial. This is a **study session, not a chat**. The latency tolerance is minutes, not milliseconds.

This changes the hardware calculus entirely. A 14B model running CPU-only on 64GB RAM at 2-6 tokens/second sounds painfully slow for interactive chat. But for generating a 2,000-word personalized guide? That's 5-15 minutes of generation time — perfectly acceptable for a "learn this new tool" workflow where the output is a document you'll study, not a quick reply you're waiting for.

The pipeline requires three capabilities that scale with model size:

1. **Comprehension** — read and understand a full GitHub README, docs page, or release notes (often 2,000-10,000 tokens of dense technical content)
2. **Context mapping** — understand your existing workflow from vault context (what tools you already use, how your setup is configured, what your skill level is)
3. **Pedagogical synthesis** — generate a personalized explanation: "Given that you use X and Y, here's how Z fits in, and here's how to set it up in your workflow"

A 135M model fails at all three — it can't comprehend long technical documents, can't reason about workflow integration, and can't generate coherent multi-paragraph explanations. A 3.8B nano handles simple tools but struggles with complex repos that have multiple configuration options and integration patterns. A 14B model hits the sweet spot: strong comprehension, solid reasoning about how pieces fit together, and good enough generation to produce a useful personalized guide.

## Evidence

**The scrape-to-learn toolchain already exists.** Multiple open-source tools solve the "URL → LLM-ready content" step:

- **Crawl4ai-RAG-with-Local-LLM**: Scrapes web docs, converts to markdown, builds embeddings for RAG with Ollama. Purpose-built for the "learn from docs" workflow.
- **OneFileLLM**: Aggregates content from GitHub repos, web pages, PDFs, YouTube transcripts into a single structured XML for LLM ingestion. Its `process_github_repo` function recursively fetches files.
- **Firecrawl**: API that converts websites into clean markdown or structured JSON — "LLM-ready output."

The scraping step is model-agnostic and lightweight. The intelligence is in what happens *after* scraping — comprehension and personalized synthesis.

**ReadMe.LLM research validates the approach.** A 2025 paper (arXiv:2504.09798) found that LLMs often fail at code generation with niche libraries even when they have access to standard documentation. They proposed LLM-oriented documentation that consistently improved performance to near-perfect accuracy. The key finding: **the documentation format matters more than the model size** for comprehension. This means the scrape step should convert docs into a structured, LLM-friendly format before feeding to the 14B model.

**14B CPU inference on 64GB RAM — the real numbers:**

| CPU | Model | Quantization | Speed | Usable? |
|-----|-------|-------------|-------|---------|
| i7-12700H | Phi-4 14B | Q4 | ~12 tok/s | Yes, comfortable |
| Modern i7/i9 | 14B generic | Q4 | ~4-8 tok/s | Yes, for document generation |
| Older i5 | 14B generic | Q4 | ~2-4 tok/s | Marginal for interactive, fine for batch |

At 6 tok/s, a 2,000-token personalized guide takes ~5.5 minutes. A 500-token quick summary takes ~1.5 minutes. This is the "go make coffee" cadence — you're not chatting, you're commissioning a document.

Memory footprint for 14B Q4 on 64GB RAM: ~12GB model + ~4GB working memory = ~16GB total. That leaves 48GB for the OS, browser with docs open, and any other tools. No memory pressure at all.

**Why specifically image/video gen tools benefit from this pipeline:**

Image and video generation tools (ComfyUI, Stable Diffusion, Kling, Runway, etc.) have three properties that make the link-to-learning pipeline especially valuable:

1. **Rapidly evolving** — new models, new nodes, new workflows every week. Your vault's knowledge goes stale fast.
2. **Complex configuration** — these tools have dozens of parameters, node connections, and workflow patterns. A simple "install and run" guide isn't enough.
3. **Workflow-dependent** — how you use ComfyUI depends on whether you're doing img2img, inpainting, video, or LoRA training. The "teach me" response needs to know your use case.

A 14B model can read a new ComfyUI custom node's README, cross-reference it with your vault's notes on your existing ComfyUI setup, and generate: "You're currently using workflow X. This new node slots in between steps 3 and 4. Here's the modified workflow. Watch out for Y because your GPU is Z."

A 3.8B model would give you a generic summary of the README. A 135M model would give you gibberish.

## Why This Matters

**The link-to-learning pipeline is the use case that justifies Tier 2.** Without it, the Mac Mini M4 16GB does everything — persistent memory, personal AI, sidecar. The PC exists specifically because:

1. **14B doesn't fit comfortably alongside the sidecar stack on 16GB** — the sidecar (micro + nano + embeddings) already uses ~9GB. Adding a 14B model at ~12GB exceeds 16GB. You'd have to stop the nano to load the 14B, breaking the always-on sidecar.

2. **CPU-only is acceptable because this isn't real-time** — you don't need a GPU for a workflow where you send a link and get a guide 5-10 minutes later. The 64GB RAM PC handles 14B inference comfortably on CPU alone.

3. **The pipeline is event-driven, not always-on** — you don't run the 14B model 24/7. You spin it up when you find a new tool, it generates the guide, the guide gets saved to your Obsidian vault, and the model unloads. Ollama handles this automatically — models load on first request and unload after idle timeout.

The full link-to-learning pipeline:

```
USER: sends URL to pipeline

STEP 1 — SCRAPE (seconds)
  Crawl4ai or OneFileLLM fetches the URL
  Converts to clean markdown
  Saves raw content to vault as reference note

STEP 2 — CONTEXTUALIZE (seconds)
  Micro (on Mac Mini) searches vault for related notes
  "What do I already know about this category of tool?"
  "What's my current setup for this workflow?"
  Retrieves relevant context notes

STEP 3 — SYNTHESIZE (5-15 minutes)
  14B model (on PC) receives: scraped content + vault context
  Generates personalized guide:
    - What this tool does (in your terms)
    - How it fits your existing setup
    - Step-by-step setup instructions (for YOUR environment)
    - What to watch out for (based on YOUR hardware/config)
  Saves guide to vault as learning note

STEP 4 — INDEX (seconds)
  Embedding index updates automatically
  Guide is now searchable in your vault
  Micro's next LoRA refresh will learn the new routing patterns
```

Notice the two-machine split: steps 1-2 happen on the Mac Mini (micro + vault, always-on), step 3 happens on the PC (14B model, on-demand), step 4 happens on the Mac Mini (embeddings, automatic). The machines communicate over the local network via Ollama's API and the shared vault (synced via Obsidian Sync, Syncthing, or a shared network folder).

## Open Thread

The pipeline assumes text-based documentation. But image/video gen tools increasingly document workflows through YouTube videos, screenshots, and visual node graphs. A text-only 14B model can't process these. Two potential solutions:

1. **Multimodal 14B model** — Qwen3.5-4B is already multimodal. A multimodal 14B (like LLaVA-Next or Qwen2.5-VL-14B) could process screenshots and visual workflows directly. This would increase memory requirements but stay within 64GB.

2. **Preprocessing step** — YouTube transcripts (via OneFileLLM) + screenshot-to-text (via a small vision model) convert visual content to text before feeding to the 14B. More complex pipeline, but works with any text model.

The multimodal angle is especially important for the user's image/video gen focus. ComfyUI workflows are inherently visual — a text description of a node graph is far less useful than seeing it. This might push the Tier 2 model choice toward a multimodal 14B rather than a text-only one.
