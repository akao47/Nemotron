# MoE Changes the Hardware Equation: Why Unified Memory Beats Discrete GPUs for Your Swarm

## The Insight

The conventional wisdom for running large local models is "buy the biggest GPU you can afford." But Nemotron 3 Super's MoE architecture (120B total / 12B active parameters) inverts this logic. Because only 12B parameters are active per inference step, the hardware strategy splits into two distinct problems: **fast memory for active parameters** and **large, cheap memory for dormant experts**. This is exactly what Apple's unified memory architecture was designed for — a single large pool where active data gets GPU-speed access and everything else sits ready to page in.

For a 3-tier swarm (super + nano + micro), the hardware assignment falls out naturally once you understand this:

**Mac Mini M4 Pro 64GB ($2,899)** — Runs the super (Nemotron 3 Super 120B-A12B at Q4 quantization needs ~60-80GB; 64GB unified memory is tight but feasible at aggressive quantization). Also runs the nano (1-4B model at ~2-4GB) and micro (50M model at <200MB) simultaneously. Unified memory means no PCIe bottleneck between CPU and GPU — all three tiers share the same memory pool. At 273 GB/s memory bandwidth, expect ~8-15 tok/s for the super model, ~45-95 tok/s for nano, and near-instant for micro.

**PC with 64GB RAM + RTX GPU** — Backup/overflow machine. Can run the super with GPU+RAM offloading (active 12B params in VRAM, dormant experts page from system RAM). With an RTX 5090 (32GB, but currently $4,000+ and near-impossible to buy), Q4_K_M fits the older Llama-3.3-Nemotron-Super-49B-v1.5 entirely in VRAM. But the RTX 5090's availability crisis (70-120% above MSRP, selling out in <30 minutes) makes this path unreliable for now.

**The existing Mac Mini M4 16GB** — Perfect dedicated host for the nano + micro. A 4B model at Q4 needs ~3GB; the 50M micro needs <200MB. Both run comfortably with room for Obsidian, Claude Code, and everything else. This frees the M4 Pro (if purchased) to dedicate its memory to the super model.

## Evidence

**Nemotron 3 Super (120B-A12B) hardware requirements** (from NVIDIA official docs and Unsloth):
- BF16 full precision: ~240GB VRAM (datacenter only)
- 4-bit quantized GGUF: 60-80GB (feasible on Mac with unified memory or PC with RAM offloading)
- Minimum RAM for GGUF with CPU offload: 64-83GB
- The MoE trick: only 12B active params per token — active layers and KV-cache can sit in fast VRAM/GPU while dormant experts page from system RAM

**Mac Mini M4 Pro specs** (from Apple):
- 14-core CPU / 20-core GPU / 16-core Neural Engine
- 64GB unified memory at 273 GB/s bandwidth
- Price: $2,899 (64GB / 2TB config)
- Thunderbolt 5 connectivity

**Apple Silicon inference benchmarks** (from community benchmarks):
- M4 Pro: Llama 3.2 1B at ~95 tok/s, 7-8B models at ~28-35 tok/s, 30B Q4 models at ~12-18 tok/s
- MLX framework is 20-30% faster than llama.cpp on Apple Silicon
- Memory bandwidth is the bottleneck for LLM inference, not compute

**RTX 5090 market reality** (March 2026):
- MSRP: $1,999 (Founders Edition only, via Verified Priority Access)
- Actual market price: $4,232 (Amazon), $3,830 (eBay used)
- Shortage is structural (GDDR7 memory scarcity from AI datacenter demand)
- Expected to persist through at least mid-2026

**Older Nemotron Super 49B (Llama-3.3 based)**:
- Q4_K_M: ~28-30GB — fits on RTX 5090 (32GB) but not RTX 4090 (24GB)
- INT4/NVFP4: ~25-30GB
- At IQ2_XXS: ~16GB (runs on consumer GPUs with quality tradeoffs)

## Why This Matters

The buy recommendation crystallizes into a simple decision tree:

1. **Cheapest viable path**: Keep existing Mac Mini M4 16GB for nano+micro. Buy Mac Mini M4 Pro 64GB ($2,899) for the super. Total additional spend: ~$2,900.

2. **Better performance path**: Mac Mini M4 Pro 64GB for nano+micro+daily work. Mac Studio M3 Ultra 192GB ($5,499+) for the super at higher quantization with room to breathe. Total additional: ~$8,400+.

3. **PC GPU path**: Add RTX 5090 to existing PC. Runs the older Nemotron Super 49B at Q4_K_M in VRAM, or the new 120B MoE with RAM offloading. But at $4,000+ actual price and terrible availability, this costs more than the Mac path for potentially less unified memory.

The Mac path wins on three fronts: **unified memory eliminates the CPU-GPU transfer bottleneck** that kills MoE performance on discrete GPUs (paging experts from RAM to VRAM across PCIe is slow); **total cost is lower** ($2,900 vs $4,000+ for a GPU you can't buy); and **the machine does double duty** as a development workstation.

The PC stays useful as a training machine later (CUDA ecosystem for fine-tuning) or as a second inference node if you cluster them.

## Open Thread

The 64GB Mac Mini M4 Pro is *tight* for the 120B MoE at Q4 — some sources say 83GB minimum. This raises two questions:
1. Can aggressive quantization (Q3_K or IQ3) bring it under 64GB without destroying quality for the specific tasks the super handles (planning, reasoning, judgment calls)?
2. Would the upcoming M5 chips with higher memory bandwidth (~153 GB/s base) meaningfully change the tok/s for this model, even at the same memory capacity?

The Mac Studio M4 Ultra (not yet announced) with 192GB+ unified memory would be the ideal super host — but timing and pricing are unknown. The M4 Pro 64GB is the pragmatic "start now" choice.
