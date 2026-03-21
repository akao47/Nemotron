# The 135M Capability Ceiling: Your Micro Is a Bouncer, Not a Bartender

## The Insight

A 135M model fine-tuned on your Obsidian vault can't be your "personal AI" in the way you're imagining — it can't explain things to you, teach you new tools, or have a meaningful conversation about your workflow. But that's not a limitation; it's the design. The micro's job is **discriminative, not generative**: classify, route, extract, tag, retrieve. It decides *what's relevant* and *where things go*. The nano (Phi-4-mini 3.8B) does the actual *thinking and explaining*.

This maps to a well-established pattern in production AI systems: 80% of tasks in an agentic pipeline are extraction, routing, or simple tool calling — exactly what a fine-tuned 100-200M model excels at. A fine-tuned 200M BERT-class model can outperform GPT-4 on classification tasks while running on CPU at sub-100ms latency. The remaining 20% — open-ended generation, reasoning, teaching — requires a larger model.

On your M4 16GB, this creates a natural two-layer personal AI:

| Layer | Model | RAM | What it does | What it can't do |
|-------|-------|-----|-------------|-----------------|
| **Micro** (always-on) | SmolLM2-135M + LoRA | ~724MB | Route queries, classify intent, tag notes, retrieve context, extract entities, trigger workflows | Explain concepts, teach tools, generate plans, reason about tradeoffs |
| **Nano** (always-on) | Phi-4-mini 3.8B Q4 | ~3GB | Explain things your way, synthesize vault content, write session summaries, review plans, teach you new tools | Deep multi-step reasoning, complex code generation, large context analysis |

Both run simultaneously on 16GB with ~7GB headroom. The micro responds in <50ms (CPU cache), the nano in 1-5 seconds. The user never talks to the micro directly — it's infrastructure. The user talks to the nano, which uses the micro as its retrieval and routing engine.

## Evidence

**SmolLM2-135M benchmarks reveal the capability ceiling:**
- HellaSwag: 42.1% (common sense reasoning — mediocre)
- ARC: 43.9% (science reasoning — mediocre)
- GSM8K: 48.2% (grade school math — barely passing)
- Function calling: 27% on Berkeley FCL (basic tool use — limited)
- Instruction following: 56.7% (follows instructions about half the time)

These numbers tell a clear story: the 135M model *understands structure* (it can follow simple instructions, call basic functions) but *can't reason* (math, science, and common sense are near-random). This is exactly the profile of a router/classifier — it can determine "this is a question about ComfyUI" and "this note belongs in the image-gen project" but it can't explain *how* to use ComfyUI.

**After fine-tuning, the picture changes for narrow tasks.** A study comparing LLMs and SLMs on requirements classification found SLMs trail LLMs by only 2% F1 score. On domain-specific classification (your vault's tags, your project categories, your workflow routing), a fine-tuned 135M model will match or beat a general-purpose 7B model — because it's been trained on *your* categories, not generic ones.

**What "trained on my information" actually means at 135M:**

| Task | Can it do it? | How well? |
|------|--------------|-----------|
| "Is this note about image gen or coding?" | Yes | Excellent after fine-tuning |
| "Tag this note with the right project" | Yes | Excellent — this is classification |
| "Find notes related to this query" | Yes | Good — routes to embedding search |
| "Summarize what I know about ComfyUI" | Barely | Poor — generation quality too low |
| "Explain how to set up a new tool" | No | Hallucination-prone, shallow |
| "Review my project plan for over-engineering" | No | Can't reason about complexity |
| "Teach me this GitHub repo's workflow" | No | Can't comprehend + synthesize + explain |

**The escalation pattern — when micro hands off to nano:**
The micro classifies every incoming request into one of these categories:
1. **Retrieval** → micro handles it (search vault, return results)
2. **Classification/tagging** → micro handles it (tag note, route to project)
3. **Generation/explanation** → escalate to nano
4. **Complex reasoning** → escalate to nano (or super, if available)

This routing decision itself is the micro's strongest capability after fine-tuning. It learns *your* escalation patterns from the LoRA adapter — which questions you always need explained vs. which you just need a quick lookup for.

**Memory budget confirms both fit comfortably on 16GB:**

```
SmolLM2-135M (inference):     ~724MB
Phi-4-mini 3.8B Q4 (inference): ~3GB
nomic-embed-text (embeddings):  ~300MB
Obsidian + OS + apps:           ~4-5GB
─────────────────────────────────────
Total:                          ~8.5-9GB
Headroom:                       ~7GB
```

The M4's unified memory architecture means no data copying between CPU and GPU — both models share the same memory pool and the M4's GPU cores accelerate inference for both simultaneously.

## Why This Matters

The "bouncer not bartender" framing resolves a common confusion about personal AI: people imagine one model that "knows everything about me and can do everything." That model would need to be 7B+ to generate useful responses, and daily retraining at 7B is impractical on consumer hardware.

The two-layer split makes daily personalization practical:
- **Personalize the cheap part** (135M micro, daily LoRA, 15 minutes, free) — this adapts routing, classification, and retrieval to your evolving vault and preferences
- **Keep the expensive part generic** (3.8B nano, no fine-tuning needed) — Phi-4-mini already knows how to explain, summarize, and reason. It doesn't need to be trained on *your* data. It just needs the micro to feed it the right context from your vault.

The nano doesn't need to know you. It needs the micro to *tell it about you* at query time. This is the RAG pattern: the micro retrieves your relevant notes, the nano reasons over them. The micro is personalized; the nano is capable. Together, they're a personal AI that knows you AND can think.

This is why Tier 1 (Mac Mini M4 16GB) handles both use cases — persistent Claude Code memory AND personal AI — simultaneously. The micro is the memory layer (always-on, personalized, cheap to update). The nano is the intelligence layer (always-on, general-purpose, no updates needed). Total footprint: ~9GB. No GPU. No cloud. No API costs.

## Open Thread

There's a gap between what the micro can classify and what the nano can generate: **structured writing**. Tasks like "update the frontmatter tags on this note" or "write a session summary in my format" are too generative for the micro but too formulaic for the nano to be the best use of its capacity. A 360M model (SmolLM2-360M, ~1.4GB) might fill this middle tier — capable of template-following and structured output without the overhead of 3.8B. But adding a third always-on model increases complexity. The simpler path is to let the nano handle it — 3.8B is overkill for tag updates, but the latency and memory cost are acceptable.

Another question: can the micro learn to route *between* Claude Code and the local nano? When you're in a Claude Code session and ask something that doesn't need Claude's capabilities (e.g., "what did I decide about the image pipeline last week?"), the micro could intercept and route to the nano + vault instead of consuming Claude API tokens. This turns the sidecar into a cost optimizer, not just a memory layer.
