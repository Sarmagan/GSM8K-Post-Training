# Llama 3.2-3B QLoRA Fine-Tuning for Math Reasoning (GSM8K)

Supervised Fine-Tuning (SFT) of Meta's Llama-3.2-3B model using QLoRA to improve mathematical reasoning capabilities on the GSM8K dataset.

## Overview

This repository contains the training pipeline for fine-tuning a Large Language Model (LLM) to solve grade-school math word problems. It leverages **Quantized Low-Rank Adaptation (QLoRA)** to efficiently train a 3-billion parameter model. The pipeline handles data preprocessing, 4-bit model quantization, parameter-efficient fine-tuning, and automated tracking/uploading to the Hugging Face Hub.

## Tech Stack

* **Core Model:** `meta-llama/Llama-3.2-3B`
* **Dataset:** `openai/gsm8k` (Grade School Math 8K)
* **Techniques:** Supervised Fine-Tuning (SFT), QLoRA, 4-bit Quantization (NF4)
* **Libraries:** `transformers`, `peft`, `trl` (SFTTrainer), `bitsandbytes`, `torch`
* **Tracking & Logging:** Weights & Biases (`wandb`)
* **Optimization:** Gradient Checkpointing, Paged AdamW, Cosine Learning Rate Scheduler

## Methodology

To make training computationally feasible, this project uses the QLoRA methodology:
1.  **Base Model Quantization:** The Llama 3.2 model is loaded in 4-bit NormalFloat (`nf4`) precision with double quantization to drastically reduce the memory footprint.
2.  **Low-Rank Adapters (LoRA):** Instead of updating all model parameters, small trainable rank decomposition matrices are injected into the Attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
3.  **Chat Templating:** The dataset is formatted using Llama 3's native special tokens (`<|start_header_id|>`, `<|eot_id|>`) to maintain consistency with the model's pre-training.

The model was trained with the following configuration to ensure stable convergence:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | 32 | Determines the size of the adapter matrices. |
| **LoRA Alpha** | 64 | Scaling factor for the LoRA weights ($2 \times r$). |
| **LoRA Dropout** | 0.1 | Prevents overfitting within the adapters. |
| **Learning Rate** | 1e-4 | Peak learning rate for the Cosine scheduler. |
| **Batch Size** | 4 | Per-device batch size (Effective Batch Size = 16 with Gradient Accumulation). |
| **Max Sequence Length**| 512 | Optimized based on the 95th percentile of tokenized GSM8K lengths. |
| **Optimizer** | `paged_adamw_32bit` | Handles memory paging during training to prevent OOM errors. |