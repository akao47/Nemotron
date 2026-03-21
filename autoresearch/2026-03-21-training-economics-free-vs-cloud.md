# The Training Economics Split: $0/Day Locally, $3/Month on RunPod, and Knowing Which Is Which

## The Insight

The most common mistake in personal AI training is trying to fine-tune when you should be using RAG, or using RAG when you should be fine-tuning. These aren't interchangeable — they solve different problems. RAG changes what the model can *see* right now. Fine-tuning changes how the model *behaves* every time. Mixing them up leads to wasted money (training on facts that change next week) or wasted effort (trying to prompt-engineer behavioral changes that need to be trained in).

For the two-tier hardware setup, this distinction maps cleanly onto a cost model:

**Tier 1 (Mac Mini, $0/day)**: LoRA fine-tune the 135M micro on behavioral patterns — your routing preferences, your project taxonomy, your escalation patterns. This is the "how I think" layer. It changes daily as your workflow evolves, and it's cheap enough to retrain every morning. RAG handles the "what I know" layer — the actual content of your vault, retrieved at query time.

**Tier 2 (PC + RunPod, ~$1-6/session)**: LoRA fine-tune the 14B model on RunPod only when you need to change how the link-to-learning pipeline *explains things to you*. Not what it knows (that's RAG), but how it teaches — your preferred explanation style, your skill level, your terminology. This happens infrequently: maybe once a month, or when you realize the model consistently misunderstands your setup.

The counterintuitive finding: **you'll spend more time deciding whether to fine-tune than actually fine-tuning.** A 7B LoRA run on RunPod costs $0.80 and takes 90 minutes. A 14B QLoRA run costs $2-6. The economic barrier isn't the cloud cost — it's knowing when training is the right move vs. just improving your RAG pipeline or prompt.

## Evidence

**RunPod pricing makes cloud fine-tuning nearly free at this scale:**

| Model | Method | GPU | Time | Cost |
|-------|--------|-----|------|------|
| 7B | LoRA | RTX 4090 ($0.39/hr) | 90 min | **$0.58** |
| 7B | QLoRA | Any 16GB+ | 1-2 hrs | **$0.39-0.78** |
| 14B | QLoRA | RTX 4090 ($0.39/hr) | 2-5 hrs | **$0.78-1.95** |
| 14B | LoRA | A100 80GB ($1.19/hr) | 2-8 hrs | **$2.38-9.52** |

RunPod bills per-second, no data transfer fees, and spot instances cut costs by 50-70%. A monthly training budget of $10 covers 2-3 fine-tuning runs of a 14B model — more than enough for iterative personalization.

**The RAG vs. fine-tune decision framework applied to your workflow:**

| What you want | Right approach | Why |
|--------------|---------------|-----|
| "Remember what I decided about the image pipeline" | RAG | Factual, vault-stored, changes over time |
| "Know my project categories and tag notes correctly" | Fine-tune micro | Behavioral — classification patterns are stable |
| "Explain new tools the way I understand them" | Fine-tune 14B | Behavioral — explanation style is consistent |
| "Know about the ComfyUI update from yesterday" | RAG | Factual, time-sensitive, retrievable |
| "Route coding questions to Claude, knowledge questions to nano" | Fine-tune micro | Behavioral — routing logic is learned |
| "Summarize this GitHub README for me" | Neither — base model capability | The 14B model already does this well |

**The critical insight from 2025-2026 practitioner guidance**: If your total knowledge base fits in ~200K tokens (roughly 150 pages of notes), you might not even need RAG. Modern models with long context windows can ingest the relevant subset directly. For a typical Obsidian vault with a few hundred notes, the nano's 128K context window (Phi-4-mini) or a 14B model's context window can hold the relevant subset without retrieval infrastructure.

This means the complexity ladder is:
1. **Start with prompt stuffing** — just paste relevant notes into context (free, zero infrastructure)
2. **Add RAG when vault grows** — embedding search when you have 500+ notes (free, local)
3. **Add LoRA when behavior needs tuning** — micro's routing, nano's style (free locally, $1-6 on RunPod)
4. **Never do full fine-tuning** — at these scales, LoRA is always sufficient

**Monthly cost model for the hybrid setup:**

| Component | Where | Frequency | Monthly cost |
|-----------|-------|-----------|-------------|
| Micro LoRA retrain (135M) | Mac Mini | Daily | **$0** |
| Embedding index update | Mac Mini | Continuous | **$0** |
| Nano inference (3.8B) | Mac Mini | Always-on | **$0** (electricity only) |
| 14B inference | PC (CPU) | On-demand | **$0** (electricity only) |
| 14B LoRA fine-tune | RunPod | Monthly | **$1-6** |
| Link-to-learning scraping | PC | On-demand | **$0** |
| **Total** | | | **$1-6/month** |

Compare this to using cloud APIs for the same workflow:
- Claude API for persistent memory: ~$10-50/month depending on usage
- GPT-4 API for link-to-learning: ~$20-100/month
- No personalization possible with cloud APIs

The local-first approach isn't just cheaper — it's the *only* way to get true personalization, because cloud APIs don't let you fine-tune on your personal data at these price points.

## Why This Matters

The economics reveal a design principle: **frequency of update should determine where training happens.**

- **Daily updates → local, free**: The micro's behavioral layer changes daily (new notes, new projects, new routing patterns). This must be free and fast, which is why it runs on the Mac Mini's MPS/MLX backend in 15 minutes.

- **Monthly updates → cloud, cheap**: The 14B's explanation style changes slowly (your preferences, skill level, and terminology evolve over weeks, not days). The $3 monthly RunPod cost is negligible, and the infrequency means you can use spot instances and be patient.

- **Never → base model**: The nano (Phi-4-mini 3.8B) doesn't need fine-tuning. It's used for general comprehension and synthesis — tasks where its base capabilities are sufficient. You personalize it through RAG (feeding it your context), not through training.

This three-frequency model means your total investment in "training your own models" is:
- **Hardware**: Mac Mini you already own + PC you're planning to buy
- **Cloud compute**: ~$3-6/month on RunPod
- **Time**: 15 minutes/day for micro retrain (automated), 2-3 hours/month for 14B fine-tune session (semi-automated)

The "training AI" dream sounds expensive. The reality, at these model sizes, is nearly free.

## Open Thread

The most expensive part isn't compute — it's **data curation**. Converting Obsidian vault notes into high-quality instruction JSONL for LoRA training requires judgment: which notes contain trainable patterns vs. which are just reference material? The nano (Phi-4-mini) can automate this conversion, but its quality determines the micro's training quality. A "garbage in, garbage out" loop where the nano generates poor training data → micro trains on it → micro routes poorly → nano gets bad context is a real risk.

The mitigation: start with manual curation for the first few training cycles (hand-pick 50-100 instruction pairs that represent your ideal behavior), then gradually let the nano take over as you validate its output quality. This is a cold-start problem, not an ongoing one.
