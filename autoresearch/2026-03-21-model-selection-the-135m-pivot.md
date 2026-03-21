# The 135M Pivot: Why Your Micro Should Be SmolLM2, Not a Custom 50M

## The Insight

The previous finding established 50M as the micro sweet spot based on capability analysis — what tasks need how much model. But capability analysis and model availability are different questions. In March 2026, there is **no production-ready, off-the-shelf 50M parameter language model**. The smallest modern models from major labs are:

- SmolLM2-135M (Hugging Face) — 135M params, ~724MB memory footprint
- Gemma 3 270M (Google) — 270M params
- SmolLM2-360M (Hugging Face) — 360M params

Building a custom 50M model means either training from scratch (feasible but requires data curation, training infrastructure, and evaluation — weeks of work for a non-developer) or distilling from a larger model (requires the Nemotron training pipeline or equivalent). Neither is a "start this weekend" path.

The pragmatic pivot: **use SmolLM2-135M-Instruct as your micro**. It's only 2.7x larger than 50M — still trivially small — and it comes pre-trained on 2 trillion tokens with instruction-following capability already baked in. At ~724MB, it runs entirely in CPU cache on any modern machine. The "always-on" requirement is effortlessly met.

For the nano tier, the landscape has converged on two clear winners at the 3-4B scale:
- **Qwen3.5-4B** — strongest general-purpose small model, natively multimodal, 262K context window, ~3GB at Q4
- **Phi-4-mini (3.8B)** — strongest at structured reasoning and code, 128K context, ~3GB at Q4

For a sidecar whose primary jobs are coding assistance, knowledge management, and plan review, **Phi-4-mini edges out** due to its reasoning and code strengths, though Qwen3.5-4B's multimodal capability and longer context are compelling for knowledge management.

For the super tier, **Nemotron 3 Super (120B-A12B)** is the clear choice — it's the newest, most capable open model, purpose-built for agentic workflows, with a 1M-token context window and configurable reasoning depth. It holds #1 on DeepResearch Bench and scores 85.6% on PinchBench for agentic tasks. Its MoE architecture means only 12B params are active per inference, keeping it runnable on consumer hardware (as established in the hardware finding).

## Evidence

**The 50M gap**: No major lab has released a model at exactly 50M parameters since the original Transformer (65M, 2017). The smallest modern options:

| Model | Params | Released | Memory (BF16) | Instruct-tuned? |
|-------|--------|----------|---------------|-----------------|
| SmolLM2-135M | 135M | Nov 2024 | ~724MB | Yes (SFT + DPO) |
| Gemma 3 | 270M | 2025 | ~540MB | Yes |
| SmolLM2-360M | 360M | Nov 2024 | ~1.4GB | Yes (SFT + DPO) |
| GPT-2 Small | 124M | 2019 | ~500MB | No |

SmolLM2-135M was trained on 2T tokens from FineWeb-Edu, DCLM, and The Stack. Post-trained with SFT on SmolTalk + DPO on UltraFeedback. Outperforms MobileLM-125M (previous SOTA at <200M) despite being trained on less data.

**Nano tier comparison** (from community benchmarks and official evaluations):

| Model | Params | HumanEval | Context | Best for |
|-------|--------|-----------|---------|----------|
| Qwen3.5-4B | 4B | ~65% | 262K | General + multimodal |
| Phi-4-mini | 3.8B | 71.3% | 128K | Reasoning + code |
| Gemma 3 4B | 4B | ~60% | 128K | Structured output |
| SmolLM3-3B | 3B | ~55% | varies | General (limited resources) |

Phi-4-mini's 71.3% HumanEval at 3.8B params is exceptional — it outperforms many 7B models on coding tasks.

**Super tier — Nemotron 3 Super (120B-A12B)**:
- Released March 11, 2026 at GTC
- #1 on DeepResearch Bench
- 85.6% on PinchBench (agentic benchmark)
- 2.2x higher throughput than GPT-OSS-120B
- 1M-token native context window
- Configurable reasoning depth (route within single model)
- Open weights, datasets, and recipes
- Already integrated by CodeRabbit, Factory, Greptile for coding agents

## Why This Matters

The model selection forms a coherent stack where each tier is purpose-matched:

```
SUPER: Nemotron 3 Super 120B-A12B
  → Orchestration, planning, complex reasoning, judgment calls
  → 1M context window = can hold entire project context
  → Only 12B active params = efficient on consumer hardware

NANO: Phi-4-mini (3.8B) or Qwen3.5-4B
  → Coding assistance, knowledge synthesis, plan review
  → Runs at 28-95 tok/s on Mac M4
  → Small enough to run alongside the super on same machine

MICRO: SmolLM2-135M-Instruct
  → Classification, routing, entity extraction, simple Q&A
  → ~724MB total footprint — runs in CPU cache
  → Always-on with zero perceptible resource impact
  → Fine-tune with LoRA for personal task specialization
```

The key trade-off for the micro: 135M is not 50M, and it's 2.7x "more model" than originally spec'd. But this is the right kind of over-engineering — you get a model that's already instruction-tuned, already capable of following structured prompts, and already fine-tunable with standard tools (LoRA on HuggingFace). The alternative (training a custom 50M from scratch) would take weeks and produce something less capable at tasks that need any instruction-following.

If the user later wants to distill down to 50M for specific micro tasks (classification, routing), the SmolLM2-135M can serve as the teacher model. Start with 135M, prove the sidecar architecture works, then distill to 50M for the tasks where 135M is overkill.

## Open Thread

The nano choice between Phi-4-mini and Qwen3.5-4B isn't fully resolved. The deciding factor may be the Obsidian integration pattern (next finding): if the nano needs to process images from notes or handle multimodal vault content, Qwen3.5-4B wins. If it's purely text-based coding and plan review, Phi-4-mini wins. Both can run on the same hardware, so the user could run both and route between them — but that adds orchestration complexity.

Another open question: can the micro (SmolLM2-135M) handle Obsidian vault search effectively? Its 135M params may be too small for semantic search over free-form notes. It might need a dedicated embedding model (like nomic-embed-text at ~137M) as a separate micro-tier component for vector search.
