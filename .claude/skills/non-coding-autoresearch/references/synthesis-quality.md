# Synthesis Quality Guide

How to distinguish genuine synthesis from dressed-up summarization.

## The Core Test

**Synthesis** creates something new from existing parts. It connects ideas, reveals patterns, or produces insights that aren't present in any single source.

**Summary** restates what's already there, possibly reorganized or condensed.

The dividing line: **Could the reader arrive at this conclusion by reading the source files you cite?**
- If they'd need to read multiple files AND think about the connections → synthesis
- If they'd get there by reading one file carefully → summary

## The Five Quality Checks

### 1. Novelty
> Does this say something that isn't stated in the source material?

**Pass**: "Nemotron's pruning pipeline implicitly performs neural architecture search — it's not just compression, it's automated design exploration with a compute budget constraint."

**Fail**: "Nemotron uses pruning to reduce model size while maintaining performance." (This is literally what pruning is.)

### 2. Connection
> Does it link two or more ideas that aren't linked in the source?

**Pass**: "The Voice RAG Agent's multi-model pipeline mirrors microservices architecture — each model is a bounded context with a single responsibility, composed through API contracts. This is the same pattern that made web services scalable, applied to inference."

**Fail**: "The Voice RAG Agent uses multiple models in a pipeline." (This is a description, not a connection.)

### 3. Transferability
> Could someone apply this insight in a different context?

**Pass**: "The SFT recipe's role-based loss masking — training only on assistant turns — is a generalizable curriculum design principle: you can shape what a model learns by controlling which tokens contribute to the loss, not just which tokens are in the training set."

**Fail**: "To use role-based loss masking, set `loss_on_user_turns=False` in the config." (This is a how-to, not a transferable insight.)

### 4. Surprise
> Would someone who read the source learn something new?

**Pass**: "The tiny_model.py provider in Super3 preserves every architectural feature at 7M params — Mamba layers, MoE routing, MTP heads. This proves architectural validity is scale-independent: if the architecture works at 7M, you can be confident the 49B version isn't carrying dead weight."

**Fail**: "The Super3 recipe includes a tiny model for testing purposes." (Obvious from reading the file.)

### 5. Specificity
> Does it make concrete claims with evidence?

**Pass**: "The GRPO alignment stage needs only relative preferences, not absolute scores. This means user feedback can be as simple as 'I preferred output A over output B' — no reward model, no scalar labels, no calibration. The barrier to personalizing alignment drops from 'train a reward model' to 'make pairwise choices.'"

**Fail**: "GRPO is a good alignment method." (Vague, no evidence, no mechanism.)

## Patterns of Good Findings

### The Hidden Design Principle
You notice a pattern repeated across multiple files that isn't documented anywhere. You name it and explain why it exists.

Example: Finding that all Nemotron recipes share a specific config inheritance pattern that implicitly creates a type system for training pipelines.

### The Cross-Domain Analogy
You recognize that a technique in the codebase is the same as a well-known pattern from a different field, and the analogy reveals something new about both.

Example: Comparing Nemotron's model tiering to microservices architecture and showing how the same scaling principles apply.

### The Implication Chain
You follow a design decision to its logical conclusions and discover implications the original authors may not have intended.

Example: Showing that Nemotron's training pipeline structure implies a "model as package" distribution model where trained models are artifacts you install.

### The Missing Piece
You identify something conspicuously absent from the codebase and reason about why it's missing and what it means.

Example: Noticing that the recipes don't include a distillation stage and reasoning about what this means for the intended use pattern.

### The Contradiction
You find two parts of the codebase that embody contradictory design philosophies and reason about which one is winning.

Example: Finding that the recipes optimize for reproducibility while the use-case examples optimize for flexibility, and analyzing the tension.

## Patterns of Bad Findings

### The Inventory
"The recipes directory contains SFT, alignment, and evaluation stages." This is a listing, not a finding.

### The Praise Report
"Nemotron's architecture is well-designed and comprehensive." This is an opinion with no insight.

### The Tutorial
"To fine-tune a Nemotron model, first prepare your data, then..." This is a how-to guide, not research.

### The Obvious Observation
"Larger models perform better but require more compute." Everyone knows this.

### The Speculation Without Evidence
"Nemotron's approach could revolutionize the industry." This is prediction without analysis.

## Revision Strategy

If a finding fails the quality check but has a kernel of insight:

1. **Identify the kernel** — what's the one sentence that's actually new?
2. **Go deeper on evidence** — read more source files to support or challenge the claim
3. **Sharpen the claim** — make it more specific and falsifiable
4. **Cut the filler** — remove everything that isn't the insight, the evidence, or the implication
5. **Re-check** — run through the five checks again

If after revision it still fails, discard it. Not every exploration yields a finding. That's normal.
