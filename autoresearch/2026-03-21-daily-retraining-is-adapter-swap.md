# Daily Retraining Is the Wrong Mental Model: It's a 10-Minute Adapter Swap, Not Training

## The Insight

When you say "train a 50M model daily on my Obsidian vault," you're imagining something much heavier than what actually needs to happen. The word "training" conjures images of GPU clusters running for days. The reality on an M4 16GB with a 135M model is this: you LoRA fine-tune a set of adapter weights (~2-4MB) on top of the frozen base model, and it takes **5-15 minutes** using Apple's MLX framework. The base model never changes. You're swapping a thin behavioral layer, not rebuilding the brain.

But here's the deeper insight: **most of what you want from "trained on my information" isn't fine-tuning at all — it's RAG (retrieval-augmented generation)**. These are two different mechanisms solving two different problems:

- **RAG** = "What do I know about X?" → searches your vault, retrieves relevant notes, feeds them as context. This is **recall**. It handles 80% of "personal AI that knows my stuff." No training required — just an embedding index over your vault that updates automatically when files change.

- **LoRA fine-tuning** = "How do I think about X?" → adapts the model's behavior to your style, terminology, preferences, and routing patterns. This is **personality**. It handles the other 20%: your shorthand, your preferred explanation style, how you organize projects, what "integrate into my workflow" means for you specifically.

The practical daily loop on your M4 16GB isn't "retrain the model." It's:

1. **Continuous (automatic)**: Embedding index updates when Obsidian vault files change (~seconds, runs in background via Smart Connections or a file watcher + nomic-embed-text)
2. **Daily (5-15 min)**: LoRA adapter refresh on SmolLM2-135M using the day's new notes converted to instruction JSONL — captures behavioral patterns, not facts
3. **Weekly (optional)**: Merge LoRA adapters into base weights if you want a clean checkpoint

## Evidence

**LoRA fine-tuning SmolLM2-135M is well-supported out of the box.** HuggingFace provides the recipe: TRL's `SFTTrainer` + PEFT's `LoraConfig` with `r=8`, `lora_alpha=16`, targeting `q_proj` and `v_proj`. The Unsloth project offers an optimized version at `unsloth/SmolLM2-135M` for even faster training. Multiple tutorials demonstrate fine-tuning SmolLM2 on custom and synthetic datasets with this exact stack.

**Apple M4 MPS training benchmarks confirm feasibility.** A research paper profiling Apple Silicon for ML training (Feng, arXiv:2501.14925) found that while MPS is ~6-10x slower than A100 for LoRA fine-tuning, Apple Silicon demonstrates superior energy efficiency. At 135M params, this speed penalty is negligible — a model this small trains fast even on slow hardware. Apple's MLX framework (built specifically for Apple Silicon) is preferred over PyTorch+MPS, eliminating MPS backend quirks entirely.

**Real-world validation**: A neurology researcher with 18 months of Obsidian lab notes extracted 412 instruction examples from her vault, fine-tuned TinyLlama-1.1B (8x larger than SmolLM2-135M) with LoRA in **22 minutes on 3 epochs**, and got domain-accurate responses to her shorthand queries. Scaling down to 135M would be proportionally faster.

**The vault-to-training-data pipeline is straightforward:**
1. Export recent/changed notes from Obsidian (markdown files)
2. Convert to instruction JSONL: `{"instruction": "...", "input": "...", "output": "..."}`
3. Keep examples concise (<256 tokens input, <128 tokens output)
4. Run LoRA fine-tune with MLX or HuggingFace PEFT
5. Swap the adapter file in Ollama (~instant, no restart needed)

LoRA adapters for a 135M model are ~2-4MB. You could store every daily adapter for a year in under 1.5GB. This means you have a **time-travel capability**: load last Tuesday's adapter to recover behavioral state from that day.

**Memory budget on M4 16GB for the full daily loop:**

| Component | RAM | Status |
|-----------|-----|--------|
| SmolLM2-135M (inference) | ~724MB | Always running |
| SmolLM2-135M (LoRA training via MLX) | ~2GB peak | 5-15 min daily |
| nomic-embed-text (embeddings) | ~300MB | Always running |
| Phi-4-mini 3.8B Q4 (nano, inference) | ~3GB | Always running |
| Obsidian + OS + Claude Code | ~4-5GB | Always running |
| **Total (inference)** | **~9GB** | **7GB headroom** |
| **Total (during daily training)** | **~11GB** | **5GB headroom** |

The M4 16GB never breaks a sweat. During the 5-15 minute daily LoRA refresh, memory peaks at ~11GB — well within budget even with everything else running.

## Why This Matters

The mental model shift from "train a model daily" to "swap a 3MB adapter daily + keep embeddings current" changes the entire feasibility calculus:

1. **No GPU needed** — MLX on M4 handles 135M LoRA in minutes. The M4's unified memory architecture means no CPU↔GPU data transfer bottleneck.

2. **No data engineering burden** — you don't need a massive dataset. 50-200 instruction examples from your recent notes is enough for a daily behavioral refresh. The facts stay in RAG.

3. **Failure is cheap** — a bad adapter swap is reverted instantly. Load yesterday's adapter. No model corruption, no retraining from scratch.

4. **The 135M ceiling is right for daily retraining** — at 350M, LoRA training on M4 takes 30-60 minutes. At 1B, it takes 2-4 hours. Daily retraining only makes sense when "daily" means "15 minutes before my first coffee." 135M is the sweet spot where the loop is fast enough to be habitual.

5. **RAG + LoRA is the right hybrid** — pure fine-tuning can't keep up with a vault that changes daily (catastrophic forgetting, stale facts). Pure RAG can't capture style and preferences (it retrieves facts but doesn't adapt behavior). The combination gives you a model that *thinks like you* (LoRA) about *what you know* (RAG).

## Open Thread

The vault-to-JSONL conversion step is the one piece that requires intelligence. Raw markdown notes aren't instruction-following examples. Someone (or something) needs to convert "my notes on ComfyUI" into `{"instruction": "How do I set up a ComfyUI workflow for img2img?", "input": "", "output": "Based on your setup, you..."}`.

This is where the nano (Phi-4-mini) earns its keep: it reads the day's changed notes, generates synthetic instruction pairs that capture the knowledge and your style, and outputs the JSONL that the micro retrains on. The nano is the teacher, the micro is the student. The nano doesn't need daily retraining because it's handling comprehension (a general capability), not personalization (which the micro's adapter handles).

Another question: should the LoRA adapter accumulate (train on all historical data each time) or be incremental (train only on today's new notes)? Accumulating prevents forgetting but gets slower over time. Incremental is fast but risks drift. The answer may be a rolling window — last 7-14 days of instruction examples, refreshed daily.
