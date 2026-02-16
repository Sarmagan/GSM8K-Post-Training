# Llama 3.2-3B QLoRA Post-training for Math Reasoning (GSM8K)

Two-stage fine-tuning of Meta's Llama-3.2-3B model: **Supervised Fine-Tuning (SFT)** followed by **Group Relative Policy Optimization (GRPO)** — using QLoRA to improve mathematical reasoning on the GSM8K dataset.

## Overview

This repository contains a training pipeline to solve grade-school math word problems. It leverages **Quantized Low-Rank Adaptation (QLoRA)** to efficiently train a 3-billion parameter model across two stages: first SFT to learn the answer format and basic reasoning, then GRPO reinforcement learning to further sharpen accuracy by rewarding correct solutions across multiple rollouts. The pipeline handles data preprocessing, 4-bit model quantization, parameter-efficient fine-tuning, and automated tracking/uploading to the Hugging Face Hub.

## Tech Stack

* **Core Model:** `meta-llama/Llama-3.2-3B`
* **Dataset:** `openai/gsm8k` (Grade School Math 8K)
* **Techniques:** Supervised Fine-Tuning (SFT), GRPO (Reinforcement Learning), QLoRA, 4-bit Quantization (NF4)
* **Libraries:** `transformers`, `peft`, `trl` (SFTTrainer, GRPOTrainer), `bitsandbytes`, `torch`
* **Tracking & Logging:** Weights & Biases (`wandb`)
* **Optimization:** Gradient Checkpointing, Paged AdamW, Cosine Learning Rate Scheduler

## Methodology

### Stage 1 — Supervised Fine-Tuning (SFT)

To make training computationally feasible, this stage uses the QLoRA methodology:
1.  **Base Model Quantization:** The Llama 3.2 model is loaded in 4-bit NormalFloat (`nf4`) precision with double quantization to drastically reduce the memory footprint.
2.  **Low-Rank Adapters (LoRA):** Instead of updating all model parameters, small trainable rank decomposition matrices are trained. 
3.  **Chat Templating:** The dataset is formatted using Llama 3's native special tokens (`<|start_header_id|>`, `<|eot_id|>`) to maintain consistency with the model's pre-training.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | 32 | Determines the size of the adapter matrices. |
| **LoRA Alpha** | 64 | Scaling factor for the LoRA weights ($2 \times r$). |
| **LoRA Dropout** | 0.1 | Prevents overfitting within the adapters. |
| **Learning Rate** | 1e-4 | Peak learning rate for the Cosine scheduler. |
| **Batch Size** | 4 | Per-device batch size (Effective Batch Size = 16 with Gradient Accumulation). |
| **Max Sequence Length**| 512 | Optimized based on the 95th percentile of tokenized GSM8K lengths. |
| **Optimizer** | `paged_adamw_32bit` | Handles memory paging during training to prevent OOM errors. |

### Stage 2 — GRPO Reinforcement Learning

Starting from the SFT adapter weights, the model is further trained with **Group Relative Policy Optimization**. Instead of learning from static labels, GRPO generates multiple candidate solutions per problem and uses reward signals to reinforce the ones that arrive at the correct answer.

**Reward functions:**
* **Correctness reward (1.0)** — Extracts the numerical answer from the model's response and compares it against the gold label. Supports multiple answer formats (GSM8K `####`, "the answer is X", last number fallback).
* **Format reward (0.2)** — Bonus for using the `#### answer` format, encouraging structured, parseable outputs.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Generations per prompt** | 8 | Number of rollouts sampled per question for group ranking |
| **Temperature** | 0.8 | Sampling temperature for diverse generations |
| **KL Penalty (β)** | 0.04 | Constrains policy drift from the reference model |
| **Learning Rate** | 3e-5 | Lower than SFT to avoid catastrophic forgetting. |
| **Target Modules** | Attention + MLP | All linear layers (`q/k/v/o_proj`, `gate/up/down_proj`) |
| **LoRA Alpha** | 32 | Standard 1:1 ratio with rank (vs. 2x in SFT) |
| **Training Samples** | 1,000 | Subset of GSM8K train split |

## Evaluation Results

The model was evaluated on the full **GSM8K test set** (1,319 samples).

| Model | Stage | Accuracy (%) | Correct Samples |
| :--- | :--- | :---: | :---: |
| Llama-3.2-3B Base | Zero-Shot | 2.58% | 34 / 1319 |
| Llama-3.2-3B-Instruct | Zero-Shot | 71.87% | 948 / 1319 |
| Llama-3.2-3B + QLoRA | SFT (1 epoch) | 32.90% | 434 / 1319 |
| Llama-3.2-3B + QLoRA | SFT (2 epochs, +MLP targets) | 43.67% | 576 / 1319 |
| Llama-3.2-3B + QLoRA | **GRPO on SFT weights** | **45.56%** | **601 / 1319** |

### Key Observations
* **SFT → GRPO pipeline works:** GRPO improved accuracy by ~2 percentage points over the best SFT checkpoint, using only 1K training samples.
* **Parameter Efficiency:** All gains were achieved by training a tiny fraction of total model parameters via QLoRA, preserving the base model's general knowledge while specializing in math.
* **No critic needed:** GRPO avoids the complexity of training a separate value network; group-relative advantages from multiple rollouts are sufficient to improve reasoning quality.