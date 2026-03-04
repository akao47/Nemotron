# Evaluator Recipes

Pre-built evaluation configurations for Nemotron models using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator).

## Overview

Evaluator recipes are standalone config files that define how to deploy and evaluate a model checkpoint. Unlike training recipes, there are no Python scripts—the CLI compiles the YAML config and passes it directly to [nemo-evaluator-launcher](https://github.com/NVIDIA-NeMo/Evaluator).

| Component | Description |
|-----------|-------------|
| `config/` | Pre-built evaluation configurations for common models |

## Available Configs

| Config | Model | Deployment | Description |
|--------|-------|------------|-------------|
| `nemotron-3-nano-nemo-ray` | Nemotron-3-Nano-30B | NeMo Framework Ray | In-framework evaluation with TP=2, EP=8 on 8 GPUs |

## Quick Start

```bash
# Evaluate with a pre-built config
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run YOUR-CLUSTER

# Override checkpoint path
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run YOUR-CLUSTER \
    deployment.checkpoint_path=/path/to/checkpoint

# Filter specific tasks
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run YOUR-CLUSTER -t adlr_mmlu

# Dry run (preview config)
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --dry-run
```

## Config Structure

Each config file has four main sections, matching the [nemo-evaluator-launcher](https://github.com/NVIDIA-NeMo/Evaluator) schema:

| Section | Purpose |
|---------|---------|
| `run` | Nemotron-specific: env.toml injection, artifact references. Stripped before passing to launcher. |
| `execution` | Where and how to run: Slurm settings, auto-export, container mounts |
| `deployment` | How to serve the model: container image, parallelism, health checks |
| `evaluation` | What to evaluate: tasks, parameters, timeouts |
| `export` | Where to send results: W&B entity/project |

The `run` section is a Nemotron extension—it provides `${run.*}` interpolations for env.toml values and is stripped during config compilation. The remaining sections pass directly to nemo-evaluator-launcher.

## Relationship to `nemotron nano3 eval`

There are two ways to run evaluations:

| Command | Use Case |
|---------|----------|
| `nemotron evaluate -c <config>` | **Generic**—standalone configs, explicit checkpoint paths, no artifact resolution |
| `nemotron nano3 eval` | **Recipe-specific**—resolves `${art:model,path}` from W&B artifact lineage, has defaults |

Both commands use the same underlying evaluation pipeline (`nemo_runspec.evaluator`). The generic command requires `-c` and an explicit config; the nano3 command has a default config with artifact integration.

## Writing Custom Configs

Use any existing config as a starting point:

```bash
# Use a custom config file
uv run nemotron evaluate -c /path/to/my-eval.yaml --run MY-CLUSTER
```

The config must have `execution`, `deployment`, `evaluation`, and `export` sections. Add a `run` section to use `${run.env.*}` interpolation from env.toml profiles.

## Further Reading

- [Evaluation Guide](../../../docs/nemotron/evaluation.md) — Full documentation
- [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) — Upstream project
- [Nano3 Evaluation](../nano3/stage3_eval/) — Recipe-specific evaluation with artifact lineage
