# Stage 2: Reinforcement Learning (RL)

This stage aligns the instruction-tuned model using GRPO (Group Relative Policy Optimization) with [NeMo-RL](../nvidia-stack.md#nemo-rl).

> **Open-Source Data Only**: This recipe uses exclusively open-sourced RL data, which is a subset of the full data used to train the released model. Results will differ from the benchmarks in the tech report. Use this recipe as a reference implementation to apply the methodology with your own data.

---

## Training Methodology

> **Training Framework**: RL alignment is implemented using [NeMo-RL](https://docs.nvidia.com/nemo/rl/latest/) with Ray for distributed actor coordination and vLLM for fast rollout generation. The Megatron backend handles distributed policy training with tensor, pipeline, context, and expert parallelism. See [NeMo-RL Documentation](https://docs.nvidia.com/nemo/rl/latest/) for implementation details.
>
> For complete methodology, see the [Nemotron 3 Super Tech Report](TBD).

### RL Pipeline Overview

The RL pipeline consists of three stages, each targeting a different alignment objective:

1. **Multi-Environment RLVR** — Unified training across 21 environments with verifiable rewards
2. **SWE-RL** — End-to-end reinforcement learning for software engineering tasks
3. **RLHF** — Principle-following generative reward model-based alignment

Multi-environment RLVR is the primary stage, training on all environments simultaneously to keep RL updates informed by the full environment mix and prevent accuracy drops across tasks. SWE-RL is handled separately because its rollouts take substantially longer and require longer context lengths. RLHF runs as a final stage to improve model behavior and interaction quality.

### Data Preparation Pipeline

Before training, the RL dataset is transformed into JSONL format compatible with NeMo-Gym:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333'}}}%%
flowchart LR
    subgraph prep["Data Preparation"]
        direction LR
        hf["HuggingFace<br/>Dataset"] --> resolve["Placeholder<br/>Resolution"]
        resolve --> jsonl["JSONL<br/>Format"]
        jsonl --> split["Train/Val/Test<br/>Split"]
    end
    split --> gym["NeMo-Gym<br/>Environment"]
    gym --> reward["Reward<br/>Computation"]

    style hf fill:#e1f5fe,stroke:#2196f3
    style resolve fill:#e1f5fe,stroke:#2196f3
    style jsonl fill:#f3e5f5,stroke:#9c27b0
    style split fill:#f3e5f5,stroke:#9c27b0
    style gym fill:#e8f5e9,stroke:#4caf50
    style reward fill:#e8f5e9,stroke:#4caf50
```

| Stage | What Happens |
|-------|--------------|
| **HuggingFace Dataset** | Load RL training blend from HuggingFace Hub |
| **Placeholder Resolution** | Resolve `_hf_placeholder` records by fetching from external datasets (DAPO, Skywork) and applying template restoration |
| **JSONL Format** | Convert to JSONL with `question`, `expected_answer`, and `responses_create_params` fields |
| **Train/Val/Test Split** | Split into training (98%), validation (1%), and test (1%) sets |
| **NeMo-Gym Environment** | Route samples to appropriate reward environments based on task type |
| **Reward Computation** | Compute verifiable rewards (math correctness, code execution, schema adherence) |

> For data preparation implementation, see **Recipe Source**: `src/nemotron/recipes/super3/stage2_rl/data_prep.py`

### GRPO Algorithm

GRPO (Group Relative Policy Optimization) optimizes the policy using group-relative advantages:

1. **Generate responses** from the current policy using vLLM
2. **Evaluate** responses using NeMo-Gym reward environments
3. **Compute group-relative advantages** across response groups per prompt
4. **Update the policy** to favor higher-reward responses with clipped gradients

The training uses an **asynchronous GRPO** setup where training and inference are decoupled across separate GPU devices. Inference workers continuously generate trajectories stored in a rollout buffer. Once enough trajectories are collected, the batch is sent to the training engine for a model update. Updated weights are pushed to inference workers as soon as a new model version is available.

**Stability Improvements:**

| Improvement | Description |
|-------------|-------------|
| **Importance Sampling Masking** | Masks importance sampling ratios from training/inference logprobs to minimize off-policy effects from policy lag |
| **In-Flight Weight Updates** | Training can update generation worker weights without waiting for ongoing rollouts to finish |
| **One-Step Off-Policy** | Inference workers are restricted to be at most one step behind the latest model version |
| **Overlong Filtering** | Excludes sequences that hit max length without EOS from loss computation |

### Multi-Environment RLVR

Training uses 21 environments across 37 datasets through NeMo-Gym:

| Environment | Description | Reward Type |
|-------------|-------------|-------------|
| **Math** | Competitive math problems (with and without Python tool) + formal proof verification | Answer correctness verification |
| **Code** | Competition code problems | Unit test pass rate |
| **STEM** | Scientific reasoning problems (including newly curated difficult problems) | Answer matching |
| **Instruction Following** | IFEval, Multi-Challenge compliance | Constraint satisfaction |
| **Safety** | Over-refusal mitigation + jailbreak robustness | Safety policy compliance |
| **Long Context** | Long-context document reasoning | Task completion |
| **Agentic Tool Use** | Conversational tool use + terminal use | Task completion |
| **Reasoning Gym** | Diverse reasoning tasks from Reasoning Gym | Task-specific rewards |

Training on all environments simultaneously provides stable gains. Single-environment training leads to severe regressions on other benchmarks.

**Data Curriculum:** Prompts where the SFT model consistently provides correct answers are filtered out. Remaining samples are sorted via a difficulty-based curriculum.

### Low-Effort Reasoning

During multi-environment RL, a subset of prompts are converted to low-effort mode. For each low-effort prompt, the reward is adjusted as a function of both correctness and the number of generated tokens.

| Phase | Scope | Proportion |
|-------|-------|------------|
| Early | Math, STEM QA, Competitive Coding | 2% of all RL prompts |
| Late | Math, STEM QA only | 1% of RL prompts |

### End-to-End SWE-RL

SWE-RL runs as a separate stage due to its distinct systems characteristics: longer rollouts, longer context, and different throughput profile.

**Key Components:**

| Component | Description |
|-----------|-------------|
| **Apptainer Containers** | Each SWE task runs in an isolated Apptainer (Singularity) container with writable tmpfs overlay |
| **OpenHands Agent Loop** | Manages the full lifecycle: initializing runtime, presenting problems, running agent steps, extracting git patches |
| **Harness Diversity** | OpenCode and Codex agent classes within OpenHands match external harness tool formats for training diversity |
| **Memory Watchdog** | Monitors aggregate RSS and proactively kills runaway processes within memory limits |
| **Command Blocklist** | Regex-based blocklist intercepts dangerous commands (killall, pkill) in shared-kernel environments |

### RLHF

The final RL stage uses a principle-following Generative Reward Model (GenRM) for RLHF:

| Parameter | Value |
|-----------|-------|
| **GenRM Initialization** | Qwen3-235B-A22B-Thinking-2507 |
| **Training Data** | HelpSteer 3 + lmarena-140k (commercially friendly subsets) + recently collected human preference data |
| **Approach** | Principle-following GenRM for guiding behavior on identity and safety domains |

The GenRM is used throughout both the multi-environment RL stage and a separate RLHF-only stage at the end of post-training.

### Hyperparameters

**GRPO Settings:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_prompts_per_step` | 256 | Prompts sampled per training step |
| `num_generations_per_prompt` | 16 | Rollouts generated per prompt |
| `train_global_batch_size` | 4096 | Batch size (single gradient update per rollout) |
| `max_generation_length` | 49K → 64K | Maximum generation length (increased during training) |
| `normalize_rewards` | true | Normalize rewards across batch |
| `val_period` | 5 | Validation every N steps |

---

## Recipe Execution

### Quick Start

<div class="termy">

```console
// 1. Prepare data (convert to JSONL format)
$ uv run nemotron super3 data prep rl --run YOUR-CLUSTER

// 2. Run RL training
$ uv run nemotron super3 rl --run YOUR-CLUSTER
```

</div>

> **Note**: The `--run YOUR-CLUSTER` flag submits jobs via [NeMo-Run](../../nemo_runspec/nemo-run.md). See [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) for setup.

### Configuration

| File | Purpose |
|------|---------|
| `config/default.yaml` | Production GRPO configuration (32 nodes) |
| `config/tiny.yaml` | Testing variant (1 node) |
| `config/data_prep/default.yaml` | Data preparation settings |
| `config/data_prep/data_blend_raw.json` | RL dataset blend |

### Data Preparation

The `data_prep.py` script converts datasets to JSONL format compatible with [NeMo-RL](../nvidia-stack.md#nemo-rl)'s NeMo-Gym interface. See [Data Preparation Module](../data-prep.md) for detailed documentation.

#### CLI Command

```bash
uv run nemotron super3 data prep rl [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via [NeMo-Run](../../nemo_runspec/nemo-run.md) |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

#### Output

```
output/stage2_rl/
├── train/
│   └── data.jsonl
├── val/
│   └── data.jsonl
├── test/
│   └── data.jsonl
└── manifest.json
```

The output is registered as a [W&B Artifact](../../nemo_runspec/artifacts.md) (`DataBlendsArtifact-rl`) for lineage tracking.

### Training

#### CLI Command

```bash
uv run nemotron super3 rl [options] [overrides...]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Attached—submits and waits, streaming logs ([NeMo-Run](../../nemo_runspec/nemo-run.md)) |
| `--batch <profile>` | Detached—submits and exits immediately ([NeMo-Run](../../nemo_runspec/nemo-run.md)) |
| `--dry-run` | Preview execution plan |
| `key=value` | Override config values ([CLI Framework](../../nemo_runspec/cli.md#dotlist-overrides)) |

#### Override Examples

```bash
# Fewer steps for testing
uv run nemotron super3 rl -c tiny grpo.max_num_steps=100

# Different temperature for generation
uv run nemotron super3 rl policy.generation.temperature=0.8

# Different learning rate
uv run nemotron super3 rl policy.megatron_cfg.optimizer.lr=5e-7

# Disable sequence packing
uv run nemotron super3 rl policy.sequence_packing.enabled=false
```

### Running with NeMo-Run

Configure execution profiles in `env.toml`:

```toml
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"

[YOUR-CLUSTER]
executor = "slurm"
account = "YOUR-ACCOUNT"
partition = "batch"
nodes = 32
ntasks_per_node = 8
gpus_per_node = 8
mem = "0"
exclusive = true
mounts = ["/lustre:/lustre"]
```

See [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) for complete configuration options.

### Artifact Lineage

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333'}}}%%
flowchart TB
    prev["ModelArtifact-sft<br/>(from Stage 1)"] --> train
    rl["RL Datasets<br/>(HuggingFace)"] --> dp["data_prep.py"]
    dp --> data["DataBlendsArtifact-rl<br/>(JSONL files)"]
    data --> train["train.py<br/>(GRPO with NeMo-RL)"]
    train --> model["ModelArtifact-rl<br/>(aligned model)"]

    style prev fill:#f3e5f5,stroke:#9c27b0
    style rl fill:#e8f5e9,stroke:#4caf50
    style dp fill:#e8f5e9,stroke:#4caf50
    style data fill:#e8f5e9,stroke:#4caf50
    style train fill:#e8f5e9,stroke:#4caf50
    style model fill:#e8f5e9,stroke:#4caf50
```

---

## Infrastructure

This stage uses the following components from the [NVIDIA AI Stack](../nvidia-stack.md):

| Component | Role | Documentation |
|-----------|------|---------------|
| [NeMo-RL](../nvidia-stack.md#nemo-rl) | Async GRPO algorithm, policy training, reward computation | [Docs](https://docs.nvidia.com/nemo/rl/latest/) |
| [NeMo-Gym](https://github.com/NVIDIA-NeMo/NeMo-Gym) | Multi-environment reward evaluation (21 environments) | [GitHub](https://github.com/NVIDIA-NeMo/NeMo-Gym) |
| [Megatron-Core](../nvidia-stack.md#megatron-core) | Distributed training primitives (TP, PP, CP, EP) | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| [Ray](https://ray.io/) | Distributed actor coordination and resource management | [Docs](https://docs.ray.io/) |
| vLLM | Fast rollout generation | [GitHub](https://github.com/vllm-project/vllm) |

### Parallelism Configuration

Training uses multiple parallelism strategies for efficient scaling:

| Parallelism | Value | Config Key |
|-------------|-------|------------|
| Tensor (TP) | 2 | `policy.megatron_cfg.tensor_model_parallel_size` |
| Pipeline (PP) | 2 | `policy.megatron_cfg.pipeline_model_parallel_size` |
| Context (CP) | 4 | `policy.megatron_cfg.context_parallel_size` |
| Expert (EP) | 8 | `policy.megatron_cfg.expert_model_parallel_size` |
| Sequence (SP) | Yes | `policy.megatron_cfg.sequence_parallel` |

**Generation (vLLM):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tensor_parallel_size` | 4 | TP for vLLM generation |
| `gpu_memory_utilization` | 0.5 | GPU memory fraction for KV cache |
| `colocated` | true | Share GPUs with training |

**Cluster:**

| Parameter | Value |
|-----------|-------|
| `num_nodes` | 32 |
| `gpus_per_node` | 8 |

### Container

```
TBD
```

---

## Next Steps

After RL completes, the aligned model can be [quantized](./quantization.md) for efficient deployment or [evaluated](./evaluate.md) against standard benchmarks.

## Reference

- [Nemotron 3 Super Tech Report](TBD) — RL methodology
- [NeMo-RL Documentation](https://docs.nvidia.com/nemo/rl/latest/) — GRPO, DPO, environments
- [NVIDIA AI Stack](../nvidia-stack.md) — NeMo-RL, Megatron-Core documentation
- [Artifact Lineage](../../nemo_runspec/artifacts.md) — W&B artifact system
- [Stage 0: Pretraining](./pretrain.md) — Pretrain the base model
- [Stage 1: SFT](./sft.md) — Instruction tuning
- [Stage 3: Quantization](./quantization.md) — Post-training quantization
- [Stage 4: Evaluation](./evaluate.md) — Benchmark evaluation
- **Recipe Source**: `src/nemotron/recipes/super3/stage2_rl/` — Implementation details
- [Back to Overview](./README.md)
