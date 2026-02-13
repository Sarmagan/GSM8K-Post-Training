import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login

# ── Configuration ──────────────────────────────────────────────────────────────
INSTRUCT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FINETUNE_BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_ADAPTER = "SArmagan/gsm8k-2026-02-13_00.29.07"  
REVISION="b05f2631f94ed55f95b0567a9ceddb117d1b82e6"

NUM_EVAL_SAMPLES = None  # Set to int for quick run, None for full test set
MAX_NEW_TOKENS = 512
EVAL_BATCH_SIZE = 32
RESULTS_FILE = "eval_results.json"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_number(text: str):
    """Extract the final numerical answer from model output."""
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return None
    return None

def load_quantized_model(model_name: str):
    """Load a model with 4-bit quantization + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generation

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer

# ── Prompt Builders ────────────────────────────────────────────────────────────
def build_prompt_instruct(question: str) -> str:
    """Prompt format for Llama-3.2-3B-Instruct."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant that solves math problems step by step. "
        "End your answer with #### followed by the final numerical answer.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def build_prompt_finetuned(question: str) -> str:
    """Prompt format matching the SFT training chat template."""
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# ── Batched Generation ─────────────────────────────────────────────────────────
def generate_batch(model, tokenizer, questions: list[str], build_prompt_fn) -> list[str]:
    """Generate answers for a batch of questions."""
    prompts = [build_prompt_fn(q) for q in questions]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    prompt_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
        )

    generated_texts = []
    for i in range(len(questions)):
        new_tokens = output_ids[i][prompt_lengths[i]:]
        generated_texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return generated_texts


# ── Evaluation Loop ────────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, test_data, label: str, build_prompt_fn):
    """Run batched evaluation and return per-sample results + accuracy."""
    correct = 0
    total = 0
    results = []

    questions = test_data["question"]
    gold_answers = test_data["answer"]
    num_samples = len(questions)
    num_batches = (num_samples + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

    for batch_idx in tqdm(range(num_batches), desc=f"Evaluating {label}"):
        start = batch_idx * EVAL_BATCH_SIZE
        end = min(start + EVAL_BATCH_SIZE, num_samples)

        batch_questions = questions[start:end]
        batch_gold = gold_answers[start:end]

        batch_generated = generate_batch(model, tokenizer, batch_questions, build_prompt_fn)

        for q, gold_text, gen_text in zip(batch_questions, batch_gold, batch_generated):
            gold_number = extract_number(gold_text)
            predicted_number = extract_number(gen_text)

            is_correct = (
                predicted_number is not None
                and gold_number is not None
                and abs(predicted_number - gold_number) < 1e-3
            )
            correct += int(is_correct)
            total += 1

            results.append({
                "question": q,
                "gold_answer": gold_number,
                "predicted_answer": predicted_number,
                "correct": is_correct,
                "generated_text": gen_text,
            })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"  {label}  —  Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"{'='*60}\n")
    return results, accuracy


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token, add_to_git_credential=True)
    else:
        login()

    # Load test data
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]
    if NUM_EVAL_SAMPLES is not None:
        test_data = test_data.select(range(min(NUM_EVAL_SAMPLES, len(test_data))))
    print(f"Evaluating on {len(test_data)} test samples  |  batch size = {EVAL_BATCH_SIZE}\n")

    # ── 1. Evaluate Instruct model ────────────────────────────────────────────
    print("Loading Instruct model …")
    instruct_model, instruct_tokenizer = load_quantized_model(INSTRUCT_MODEL)
    instruct_results, instruct_acc = evaluate_model(
        instruct_model, instruct_tokenizer, test_data,
        label="Llama-3.2-3B-Instruct",
        build_prompt_fn=build_prompt_instruct,
    )

    del instruct_model
    torch.cuda.empty_cache()

    # ── 2. Evaluate Fine-tuned model (LoRA adapter on base) ──────────────────
    print("Loading fine-tuned model (base + LoRA adapter) …")
    ft_model, ft_tokenizer = load_quantized_model(FINETUNE_BASE_MODEL)
    ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_ADAPTER, revision=REVISION)
    ft_model.eval()
    ft_results, ft_acc = evaluate_model(
        ft_model, ft_tokenizer, test_data,
        label="Fine-Tuned Model",
        build_prompt_fn=build_prompt_finetuned,
    )

    del ft_model
    torch.cuda.empty_cache()

    # ── 3. Summary & Save ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Instruct Model Accuracy:   {instruct_acc:.2%}")
    print(f"  Fine-Tuned Model Accuracy: {ft_acc:.2%}")
    print(f"  Improvement:               {ft_acc - instruct_acc:+.2%}")
    print("=" * 60)

    output = {
        "timestamp": datetime.now().isoformat(),
        "instruct_model": INSTRUCT_MODEL,
        "finetune_base_model": FINETUNE_BASE_MODEL,
        "finetuned_adapter": FINETUNED_ADAPTER,
        "num_eval_samples": len(test_data),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "instruct_accuracy": instruct_acc,
        "finetuned_accuracy": ft_acc,
        "improvement": ft_acc - instruct_acc,
        "instruct_results": instruct_results,
        "finetuned_results": ft_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_FILE}")

    # ── 4. Sample comparisons ─────────────────────────────────────────────────
    print("\n── Sample Comparisons ──────────────────────────────────────")
    for i in range(min(5, len(test_data))):
        inst = instruct_results[i]
        ft = ft_results[i]
        print(f"\nQ: {inst['question'][:120]}...")
        print(f"  Gold:     {inst['gold_answer']}")
        print(f"  Instruct: {inst['predicted_answer']}  {'✓' if inst['correct'] else '✗'}")
        print(f"  FineTune: {ft['predicted_answer']}  {'✓' if ft['correct'] else '✗'}")

if __name__ == "__main__":
    main()