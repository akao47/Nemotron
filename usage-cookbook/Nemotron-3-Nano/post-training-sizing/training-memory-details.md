# Training Memory Estimation Details: Nemotron 3 Nano 30B-A3B

> Detailed derivations and activation scaling for
> [NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
> For quick GPU counts see [Quick Sizing Rules](./quick-sizing-rules.md).

---

## Model Architecture Summary

| Property | Value |
|----------|-------|
| Total parameters | 31.6B |
| Active parameters per token | 3.5B |
| Precision | BF16 |
| Architecture | Hybrid Mamba-2 + MoE + Attention |
| Layers | 52 (23 Mamba-2 + 23 MoE + 6 Attention) |
| Hidden size | 2688 |
| Vocab size | 131,072 |
| Experts per MoE layer | 128 routed + 1 shared |
| Experts activated per token | 6 |
| Max supported context | 1,048,576 (1M); default config: 262,144 (256K) |

### Parameter Breakdown

| Component | Params | BF16 Memory | % of Total |
|-----------|-------:|------------:|-----------:|
| Routed experts (128/layer × 23 MoE layers) | 29.37B | 54.7 GiB | 93.0% |
| Mamba-2 layers (23) | 0.89B | 1.7 GiB | 2.8% |
| Embeddings + LM head | 0.70B | 1.3 GiB | 2.2% |
| Shared experts (23 layers) | 0.46B | 0.9 GiB | 1.5% |
| Attention layers (6) | 0.14B | 0.3 GiB | 0.4% |
| Routers + norms | 0.01B | 0.02 GiB | <0.1% |
| **Total** | **31.58B** | **58.8 GiB** | |

Non-expert vs. expert split:

| Category | Params | BF16 Memory |
|----------|-------:|------------:|
| Non-expert (replicated or sharded by framework) | 2.20B | 4.1 GiB |
| Routed experts (split by EP) | 29.37B | 54.7 GiB |

---

## Full Supervised Finetuning (SFT) memory estimation

The **16 bytes/param** is the minimum **static** memory at BF16 precision — fixed regardless of
sequence length or batch size:

| Item | Bytes/Param | Notes |
|------|:-----------:|-------|
| Weight (BF16) | 2 | The model itself |
| Gradient (BF16) | 2 | Backward pass output |
| Optimizer (Adam; FP32) | 12 | Full-precision weights, first and second moments |
| **Static per-param total** | **16** | |

On top of that, **activation memory** (dynamic) must be stored for the backward
pass. At short sequences (≤2K tokens) activations are a small fraction of total
memory and static costs dominate. As sequence length grows, activations scale
roughly linearly and become the dominant consumer beyond ~8K tokens — at that
point they may require additional parallelism (TP or CP) on top of what the
static costs already need. See [Activation Memory & Sequence Length](#activation-memory--sequence-length)
for scaling details.

With a distributed optimizer the 12 bytes of Adam state are sharded across DP ranks,
giving **4 + 12/DP_size bytes/param** per GPU (plus activations).

---

## LoRA memory estimation

For dense models, a common heuristic is "LoRA ≈ 2× model size." For this MoE model
the multiplier is much lower because only 3.5B of 31.6B params are active per token,
keeping activations small relative to total model size.

| Item | Size | Notes |
|------|------|-------|
| Frozen weights (BF16) | 2 bytes/param → 58.8 GiB | Dominant cost; unchanged by LoRA |
| LoRA adapter weights | Rank-dependent | Rank 8 → 0.4 GiB, rank 32 → 1.6 GiB |
| Optimizer states | 12 bytes/trainable param | Only over adapter params — small at low ranks |
| Activations | Sequence-length dependent | See [Activation Memory & Sequence Length](#activation-memory--sequence-length) |
| **Static total** | **~62 GiB (rank 8) to ~72 GiB (rank 32)** | Frozen weights + adapter + optimizer + gradients |

The base model footprint doesn't change — LoRA only saves on the optimizer side.
Activation memory is additional and scales the same way as in full SFT.
For per-module trainable parameter counts, see
[LoRA Trainable Parameter Derivation](./lora-trainable-param-derivation.md).

---

## Expert Parallelism — Why It Matters Here

93% of this model's 31.6B parameters live in 128 routed experts (29.37B params,
54.7 GiB in BF16). Expert Parallelism (EP) is therefore the **first knob to tune**:

- **Expert Parallelism (EP)** distributes whole experts across GPUs — at EP=8,
  each GPU holds 16 of 128 experts (~6.8 GiB) instead of all 128 (~54.7 GiB).
- **Tensor Parallelism (TP)** splits individual weight matrices across GPUs.
  Non-expert weights are only 2.2B (4.1 GiB), so TP>1 saves little weight
  memory at short sequences while adding inter-GPU communication overhead.
  Use TP when you need to split activations for longer sequences (>8K tokens).
- **Sequence Parallelism (SP)** reduces activation memory within TP groups by
  partitioning sequence-dimension activations across TP ranks. SP requires
  TP>1; by itself it does not reduce weight memory.
- **Context Parallelism (CP)** shards long sequences across GPUs and is useful
  when TP alone is not enough at very long context. CP primarily targets
  activation memory, not weight memory.
- **Pipeline Parallelism (PP)** splits layers across GPUs — rarely needed here;
  PP=1 in all shipped recipes.
- **DTensor (FSDP2)**, used in Automodel and NeMo RL, additionally shards
  non-expert weights and optimizer states across data-parallel ranks. When
  applicable, this can help run smaller GPU counts than EP-only layouts.

**Rule of thumb:** set EP first to fit expert weights; then scale for long
context with TP (+SP) and CP; use DTensor (FSDP2) when available to reduce
replicated non-expert/optimizer memory. Lower EP values (1, 2, 4) can work as
overrides if memory permits, but shipped recipes use EP=8 for headroom.

---

## Activation Memory & Sequence Length

### Activation Memory Scaling

Weight memory is fixed regardless of sequence length. Activation memory scales
approximately **linearly** with sequence length and becomes the dominant consumer
beyond ~8K tokens. The linear approximation holds further than for dense
transformers because:

- **23 Mamba-2 layers** contribute less activation memory growth than attention layers — they use a fixed-size recurrent state rather than a KV cache, and only store chunk-boundary states during training.
- **23 MoE layers** scale linearly (activations per expert per token).
- **Only 6 attention layers** are quadratic, but Flash Attention reduces them to linear memory.

| Regime | Seq Length | Bottleneck | What to Tune |
|--------|-----------|------------|--------------|
| **Weight-bound** | < 8K | Frozen expert weights | EP (more GPUs = fewer experts/GPU) |
| **Activation-bound** | > 8K | Forward-pass activations | TP, CP, MBS, gradient checkpointing |

Doubling sequence length or micro-batch size roughly doubles activation memory.
TP and CP divide activations across GPUs proportionally.

**Example** (8× H100, Megatron-Bridge, EP=8, LoRA rank=32, static ~15 GiB/GPU):

- **seq=2K, MBS=1, TP=1:** activations are a small fraction of 80 GiB — fits
  comfortably with room for larger MBS.
- **seq=16K, MBS=1, TP=1:** activations grow ~8× relative to 2K. Approaches the
  80 GiB limit at MBS=1 — this is roughly the max sequence length on a single
  8-GPU node without additional parallelism.
- **seq=32K+:** exceeds 80 GiB at TP=1. Add TP=2 (16 GPUs) or TP=4 (32 GPUs)
  to halve or quarter activations per GPU. CP can further divide across GPUs for
  very long context (64K+).

> Min GPUs = EP × TP × CP. TP splits both activations and non-expert weights
> across GPUs. Expert weights remain split by EP only.

#### FP8 KV cache for extended context

FP8 quantization of the KV cache can approximately **halve KV cache memory**
with minimal accuracy degradation. While attention layers are only 6 of 52,
the KV cache is the dominant attention activation cost at longer sequences
and larger micro-batch sizes, making FP8 most impactful there. Available in
both Megatron-Bridge (via Transformer Engine) and Automodel.

### Mamba-2 and Activation Memory

23 of 52 layers are Mamba-2. During inference, they use a fixed-size recurrent
state instead of a KV cache — giving O(1) memory scaling with sequence length.
During training, input activations are still O(seq_len) per layer, but Mamba-2
avoids the additional KV cache overhead that attention layers incur, storing only
chunk-boundary recurrent states. This is why overall activation memory grows more
slowly than for dense transformers.

---

## Mamba-2 Fused Kernel Constraint on LoRA Targets

The Mamba-2 implementation passes raw weight tensors directly into fused CUDA
kernels (e.g., `mamba_split_conv1d_scan_combined`). When LoRA wraps these modules,
the standard `forward()` is **never called** — the kernel reads `.weight` directly:

1. **LoRA adapters on these modules produce zero gradients** (no learning).
2. **`merge_and_unload()` will fail** with shape mismatch errors.

**Safe LoRA targets:**

| Module | Layer Types | Safe? |
|--------|-------------|:-----:|
| `linear_qkv` | Attention | Yes |
| `linear_proj` | Attention | Yes |
| `linear_fc1` | MoE experts, shared expert | Yes |
| `linear_fc2` | MoE experts, shared expert | Yes |
| `in_proj` | Mamba-2 | Yes |
| `out_proj` | Attention: Yes, **Mamba-2: NO** | Use `exclude_modules` to filter Mamba-2 |
| `conv1d` | Mamba-2 | **NO** — fused kernel |

Framework exclusion settings:
- **Automodel:** `exclude_modules: ["*.out_proj"]`
- **NeMo RL:** `exclude_modules: ['*out_proj*']`
- **Megatron-Bridge:** the PEFT recipe includes `"out_proj"` in `target_modules`
  without an `exclude_modules` filter — verify that the framework filters by layer
  type internally, or add an explicit exclusion.

---

## Non-Expert Weight Distribution

- **Megatron-Bridge (EP only):** non-expert weights (4.1 GiB) are **replicated** on
  every GPU. EP splits only the routed experts.
- **Automodel / NeMo RL (DTensor + EP):** non-expert weights are **sharded** across
  DP ranks via FSDP2, in addition to experts being split by EP.

This affects the minimum GPU count for Full SFT (8 GPUs for Automodel vs 16 for
Megatron-Bridge). The trade-off is that replication avoids the communication overhead
of weight gathering during forward/backward passes, which can improve training
throughput at scale.

---

## References

### Documentation
- [Nemotron 3 Nano (Megatron-Bridge)](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html)
- [HuggingFace model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Automodel fine-tuning guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm-finetuning.html)
- [NeMo RL SFT guide](https://docs.nvidia.com/nemo/rl/latest/guides/sft.html)
- [NeMo RL GRPO guide](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html)

### Recipes and Configs
- [Megatron-Bridge recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py)
- [Automodel PEFT config](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml)
- [NeMo RL SFT config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2.yaml)
- [NeMo RL GRPO config (Full)](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2.yaml)
- [NeMo RL GRPO config (LoRA)](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-lora.yaml)