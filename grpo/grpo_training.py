import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
import wandb
import trl

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed, GenerationConfig
from peft import LoraConfig, PeftModel, get_peft_model
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from huggingface_hub import login
from typing import Optional, List
from google.colab import userdata
from peft import prepare_model_for_kbit_training

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "gsm8k-grpo"
HF_USER = "SArmagan"

FINETUNED_ADAPTER = "SArmagan/gsm8k-2026-02-13_18.57.41"
REVISION = "2ebf79024676a52174b24c567a0322268d20287d"

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Overall hyperparameters
EPOCHS = 1
BATCH_SIZE = 16 # 8
GRADIENT_ACCUMULATION_STEPS = 2 # 8

# QLoRA hyperparameters
LORA_R = 32
LORA_ALPHA = LORA_R  # Standard ratio; previously 2x which effectively doubles LR for LoRA params
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.1

# Training hyperparameters
LEARNING_RATE = 3e-5 #1e-4
LR_SCHEDULER_TYPE = 'cosine'
WARMUP_RATIO = 0.1

# GRPO specific settings
NUM_GENERATIONS = 8 # 8
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 512

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking
LOG_STEPS = 1
SAVE_STEPS = 5
LOG_TO_WANDB = True


def gsm8k_dataset_load():
    dataset = load_dataset("openai/gsm8k", "main")

    training_data = dataset["train"]
    test_data = dataset["test"]

    print("example training question: ", training_data[0]["question"])
    print("example training answer: ", training_data[0]["answer"])

    print("number of training samples: ", len(training_data))
    print("number of test samples: ", len(test_data))

    return training_data, test_data


def load_quantized_base(model_name: str):
    """Load the base model with 4-bit quantization + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config,
    )
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Extract the numerical answer from model's response.

    Looks for numbers in different formats:
    - #### 42 (GSM8K format)
    - "The answer is 42"
    - "= 42"
    - Falls back to the last number in the text
    """

    # First, check for GSM8K format (#### answer)
    if "####" in text:
        answer = text.split("####")[-1].strip()
        answer = answer.replace(',', '').replace('$', '')
        try:
            return float(answer)
        except:
            pass

    # Try various answer patterns
    patterns = [
        r"(?:The answer is|answer:|Answer:)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:equals?|=)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:total|sum|result|Total|Final answer)\s*(?:is|:|=)?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"Therefore,?\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                # FIX: matches is a list, take the last match
                answer = matches[-1].replace(',', '').replace('$', '')
                return float(answer)
            except:
                continue

    # Last resort: find the last number in the text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass

    return None


def extract_gold_answer(answer_text: str) -> Optional[float]:
    """Extract the numerical answer from GSM8K's gold answer field."""
    return extract_numerical_answer(answer_text)


def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Reward function compatible with TRL GRPOTrainer.

    TRL passes dataset columns as keyword arguments matching column names.
    Since we renamed 'question' -> 'prompt', the 'answer' column is passed as the 'answer' kwarg.

    Returns raw 0/1 rewards — let GRPOTrainer handle advantage normalization.
    """
    rewards = []
    for prompt, completion, gold_answer in zip(prompts, completions, answer):
        # Extract the completion text
        if isinstance(completion, list):
            # completion is a list of message dicts
            text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        correct_answer = extract_gold_answer(gold_answer)

        if correct_answer is None:
            rewards.append(0.0)
            continue

        predicted = extract_numerical_answer(text)

        if predicted is not None and abs(predicted - correct_answer) < 0.01:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward(prompts, completions, **kwargs):
    """
    Bonus reward for responses that use the #### format for their final answer.
    Encourages the model to produce parseable outputs.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        if "####" in text:
            rewards.append(0.2)
        else:
            rewards.append(0.0)

    return rewards


def GRPO_with_QLoRA():

    ft_model, tokenizer = load_quantized_base(BASE_MODEL)
    # ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_ADAPTER, revision=REVISION)
    # ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_ADAPTER, revision=REVISION, is_trainable=True, adapter_name="sft")
    ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_ADAPTER, revision=REVISION, is_trainable=True)

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
    )

    train, test = gsm8k_dataset_load()
    train_dataset = train.rename_columns({
        'question': 'prompt',
    })
    eval_dataset = test.rename_columns({
        'question': 'prompt',
    })

    train_dataset = train_dataset.shuffle(seed=42).select(range(1000)) # uusing 1000 training examples
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(50))

    # # Merge the existing adapter weights, then apply a fresh trainable LoRA
    # ft_model = ft_model.merge_and_unload()
    # ft_model = prepare_model_for_kbit_training(ft_model)

    # ft_model.print_trainable_parameters()

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ft_model.generation_config = GenerationConfig(
        temperature=TEMPERATURE,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_id],
    )

    # ## TEST EXAMPLE ##
    # prompt = "What is 2 + 3?"
    # messages = [{"role": "user", "content": prompt}]
    # input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # print(input_text)
    # inputs = tokenizer(input_text, return_tensors="pt").to(ft_model.device)

    # with torch.no_grad():
    #     output = ft_model.generate(**inputs, max_new_tokens=200, temperature=0.8, do_sample=True)
    # print(tokenizer.decode(output[0], skip_special_tokens=False))
    # ## TEST EXAMPLE ##

    # lora_config = LoraConfig(
    #     r=LORA_R,
    #     lora_alpha=LORA_ALPHA,
    #     target_modules=TARGET_MODULES,
    #     lora_dropout=LORA_DROPOUT,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # ft_model = get_peft_model(ft_model, lora_config)
    # ft_model.print_trainable_parameters()

    # ft_model.add_adapter("grpo", lora_config)
    # ft_model.set_adapter(["sft", "grpo"])  # Both active: SFT frozen, GRPO trainable
    # ft_model.print_trainable_parameters()

    GRPO_config = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,

        output_dir=PROJECT_RUN_NAME,
        run_name=RUN_NAME,
        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,

        save_steps=SAVE_STEPS,

        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,

        beta = 0.04,

        report_to="wandb" if LOG_TO_WANDB else None,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="no",
        eval_steps=SAVE_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=LOG_STEPS,
        save_total_limit=10,

        bf16=True,
        fp16=False,
        bf16_full_eval=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=ft_model,
        reward_funcs=[correctness_reward, format_reward],
        args=GRPO_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")

    wandb.finish()

if __name__ == "__main__":
    print("Visible GPUs:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print(f"TRL version: {trl.__version__}")
    GRPO_with_QLoRA()