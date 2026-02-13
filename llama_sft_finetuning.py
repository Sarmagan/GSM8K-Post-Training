import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed, GenerationConfig
from peft import LoraConfig, PeftModel, get_peft_model
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from huggingface_hub import login

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "gsm8k"
HF_USER = "SArmagan"

RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Overall hyperparameters 
EPOCHS = 1 
BATCH_SIZE = 4 
MAX_SEQUENCE_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 4

#  QLoRA hyperparameters 
LORA_R = 32 
LORA_ALPHA = LORA_R * 2
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS 
LORA_DROPOUT = 0.1

# Training hyperparameters 
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = 'cosine'
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking
VAL_SIZE = 500 
LOG_STEPS = 5 
SAVE_STEPS = 100 
LOG_TO_WANDB = True

def gsm8k_dataset_load():
    dataset = load_dataset("openai/gsm8k", "main")

    training_data = dataset["train"]
    test_data = dataset["test"]

    print("example training question: ", training_data[0]["question"])
    print("example training answer: ", training_data[0]["answer"])

    print("number of training samples: ", len(training_data))
    print("number of test samples: ", len(test_data))

    # question = training_data[0]["question"]

    return training_data, test_data

def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"   
   
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")
    return model, tokenizer

def token_table(prompt, tokenizer):
    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)

    # Decode each token individually to get its representation
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]

    # Create the token table
    column_width = 21
    # Create and print the header
    header = f"{'Token Ids':^{column_width}}" + f"{'Token Strings':^{column_width}}"
    print(header)
    print("=" * len(header))
    # Print the tokens row by row
    for token_id, token_string in zip(token_ids, token_strings):
        # Format token string with quotes
        token_string_quoted = f"'{token_string}'"
        print(f"{token_id:^{column_width}}" + f"{token_string_quoted:^{column_width}}")

def model_output(prompt, model, tokenizer):
    ids = tokenizer(prompt, return_tensors='pt').to(model.device)

    outputs = model.generate(
        input_ids=ids['input_ids'],
        attention_mask=ids['attention_mask'],
        max_new_tokens=5)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Model output: {outputs}')
    print(f'Model decoded output: {decoded}\n')
    token_table(decoded, tokenizer)

def main():
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)

    question = 'The integral of x^2 from 0 to 2 is'
    # token_table(question, tokenizer)

    model_output(question, model, tokenizer)

def credentials():
    # Option 1: Set these as environment variables before running:
    #   export HF_TOKEN=your_token
    #   export WANDB_API_KEY=your_key
    # Option 2: They'll be read from ~/.huggingface/token and ~/.netrc if you've logged in before

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=True)
    else:
        login()  # Will prompt interactively or use cached token

    # wandb_api_key = os.environ.get("WANDB_API_KEY")
    # if wandb_api_key:
    #     wandb.login(key=wandb_api_key)
    # else:
    #     wandb.login()  # Will prompt interactively or use cached key

    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

def SFT_with_QLoRA():
    credentials()
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)

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
        'answer': 'completion'
    })
    eval_dataset = test.rename_columns({
        'question': 'prompt', 
        'answer': 'completion'
    })

    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    train_parameters = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb" if LOG_TO_WANDB else None,
        run_name=RUN_NAME,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    fine_tuning = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_parameters,
        args=train_parameters
    )

    torch.cuda.empty_cache()

    fine_tuning.train()
    # Push our fine-tuned model to Hugging Face
    fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")

    wandb.finish()

#  def fine_tuned_model():
#     pass

def max_sequence_length_cutoff():
    train, test = gsm8k_dataset_load()
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)
    
    lengths = [len(tokenizer.encode(q + " " + a)) for q, a in zip(train["question"], train["answer"])]
    print(f"Mean: {np.mean(lengths):.0f}, Max: {np.max(lengths)}, 95th percentile: {np.percentile(lengths, 95):.0f}")
    # Mean: 159, Max: 458, 95th percentile: 269

if __name__ == "__main__":
    # main()
    SFT_with_QLoRA()
    # max_sequence_length_cutoff()