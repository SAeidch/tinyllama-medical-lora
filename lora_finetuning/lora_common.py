"""
lora_common.py - settings and helpers shared by train_lora.py and evaluate_lora.py

Keeping the prompt format and the tokenization in ONE place guarantees that the
model is trained, validated and tested on exactly the same kind of text.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Models, data and prompt
# --------------------------------------------------------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"                  # base model (Hugging Face)
DATASET_NAME = "medalpaca/medical_meadow_medical_flashcards"       # Medical Meadow flashcards
SEED = 42

# One fixed system message, used for training AND testing
SYSTEM_PROMPT = "You are a helpful medical assistant. Answer the question accurately and concisely."


def build_prompt(tokenizer, question):
    """
    Turn a question into a TinyLlama instruction prompt using the model's own chat template
    (<|system|> ... </s> <|user|> ... </s> <|assistant|>). The returned text ends exactly
    where the model should start writing its answer.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, question, answer, max_length):
    """
    Build one training example: prompt + answer + end-of-sequence token.
    `labels` are -100 (ignored by the loss) for the prompt tokens, so the model is only
    trained to produce the ANSWER. Returns None if no answer token fits in max_length.
    """
    prompt = build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    full_ids = full_ids[:max_length]                       # truncate long examples
    n_prompt = min(len(prompt_ids), len(full_ids))
    if n_prompt >= len(full_ids):                          # the answer was cut off completely
        return None
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype():
    """bf16 on GPUs that support it (Ampere or newer), fp16 on older GPUs, fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


def get_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:                              # needed for batching
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name, dtype=None):
    """Load the base causal LM (the `dtype` keyword is called `torch_dtype` in older transformers)."""
    dtype = dtype or pick_dtype()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(get_device())
