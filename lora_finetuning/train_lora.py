"""
train_lora.py - Problem 3, program 1: LoRA fine-tuning of TinyLlama on Medical Meadow flashcards

Steps (matching the assignment):
  1. Download the dataset from Hugging Face, inspect it and preprocess the question-answer records
  2. Convert each record into a TinyLlama instruction-response prompt (chat template)
  3. Apply LoRA adapters to the transformer layers of TinyLlama
  4. Fine-tune the adapters with a 90% / 10% training / validation split
  5. Save the adapter (and the training metrics)

Run one configuration at a time (outputs go to runs/<config name>/):
    python train_lora.py --config baseline
    python train_lora.py --config rank16
    python train_lora.py --config highlr
Use --max-samples N to train on a random subset (the whole dataset has about 34,000 records,
which takes many hours on a laptop GPU). Use the SAME --max-samples for every configuration,
so that all configurations see the same training and validation data.
Quick smoke test:  python train_lora.py --config baseline --max-samples 200 --epochs 1
"""

import argparse
import inspect
import json
import random
import re
import shutil
import statistics
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

from lora_common import (DATASET_NAME, MODEL_NAME, SEED, encode_example, get_device, get_tokenizer,
                         load_base_model, pick_dtype, save_jsonl)

# --------------------------------------------------------------------------
# Training configurations. "baseline" is the configuration suggested in the assignment;
# the other two change ONE thing each (LoRA capacity, learning rate).
# --------------------------------------------------------------------------
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # the attention projections of every layer

CONFIGS = {
    "baseline": dict(rank=8, alpha=16, lr=2e-4, epochs=3, batch_size=2, grad_accum=4, max_length=512),
    "rank16": dict(rank=16, alpha=32, lr=2e-4, epochs=3, batch_size=2, grad_accum=4, max_length=512),
    "highlr": dict(rank=8, alpha=16, lr=5e-4, epochs=3, batch_size=2, grad_accum=4, max_length=512),
}


# --------------------------------------------------------------------------
# Step 1: load, inspect and preprocess the data
# --------------------------------------------------------------------------
def load_records(source):
    """`source` is a Hugging Face dataset name, or a local .json/.jsonl file (used for offline tests)."""
    if source.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=source, split="train")
    return load_dataset(source, split="train")


def clean_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def preprocess(dataset):
    """
    Turn raw records (instruction / input / output) into {"question", "answer"} pairs.
    The `input` field holds the question and `output` the answer; the `instruction` field is the
    same generic sentence in every record, so it is replaced by our own system prompt.
    Records with an empty question or answer and exact duplicates are removed.
    """
    rows, seen, removed = [], set(), {"empty_question": 0, "empty_answer": 0, "duplicate": 0, "too_long_question": 0}
    for rec in dataset:
        q, a = clean_text(rec.get("input")), clean_text(rec.get("output"))
        if not q:
            removed["empty_question"] += 1
        elif not a:
            removed["empty_answer"] += 1
        elif len(q) > 1500:
            removed["too_long_question"] += 1
        elif (q.lower(), a.lower()) in seen:
            removed["duplicate"] += 1
        else:
            seen.add((q.lower(), a.lower()))
            rows.append({"question": q, "answer": a})
    return rows, removed


def inspect_data(dataset, rows, removed, tokenizer, max_length):
    """Print (and return) a short report on the dataset: size, examples, length statistics."""
    def stats(values):
        values = sorted(values)
        return {"mean": round(statistics.mean(values), 1), "median": statistics.median(values),
                "p95": values[int(0.95 * (len(values) - 1))], "max": values[-1]}

    token_lens = [len(tokenizer(r["question"] + " " + r["answer"])["input_ids"]) for r in rows]
    report = {
        "raw_records": len(dataset), "columns": list(dataset.column_names),
        "removed_during_cleaning": removed, "clean_records": len(rows),
        "question_words": stats([len(r["question"].split()) for r in rows]),
        "answer_words": stats([len(r["answer"].split()) for r in rows]),
        "question_plus_answer_tokens": stats(token_lens),
        "share_longer_than_max_length": round(sum(t > max_length for t in token_lens) / len(token_lens), 4),
    }
    print("\n=== Dataset inspection ===")
    print(json.dumps(report, indent=2))
    print("Example records:")
    for r in random.Random(SEED).sample(rows, min(3, len(rows))):
        print(f"  Q: {r['question']}\n  A: {r['answer']}\n")
    return report


# --------------------------------------------------------------------------
# Steps 3 and 4: LoRA + fine-tuning
# --------------------------------------------------------------------------
def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def make_training_args(output_dir, cfg, use_bf16, use_fp16):
    """
    Build TrainingArguments. Some argument names differ between transformers versions
    (eval_strategy / evaluation_strategy, warmup_ratio / warmup_steps), so the names that the
    installed version supports are looked up instead of being hard-coded.
    """
    supported = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = dict(
        output_dir=str(output_dir), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        lr_scheduler_type="cosine", weight_decay=0.0,
        logging_steps=10, save_strategy="no", report_to="none", seed=SEED,
        bf16=use_bf16, fp16=use_fp16,
    )
    kwargs["eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"] = "epoch"   # evaluate every epoch
    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = 0.03          # 3% of the steps for learning-rate warm-up
    else:
        kwargs["warmup_steps"] = 0.03          # newer versions: a value below 1 means a fraction of all steps
    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of TinyLlama on Medical Meadow flashcards.")
    parser.add_argument("--config", default="baseline", choices=list(CONFIGS), help="named configuration")
    parser.add_argument("--name", default=None, help="run name (default: the config name)")
    parser.add_argument("--rank", type=int, help="override LoRA rank")
    parser.add_argument("--alpha", type=int, help="override LoRA alpha")
    parser.add_argument("--lr", type=float, help="override learning rate")
    parser.add_argument("--epochs", type=int, help="override number of epochs")
    parser.add_argument("--batch-size", type=int, help="override per-device batch size")
    parser.add_argument("--grad-accum", type=int, help="override gradient accumulation steps")
    parser.add_argument("--max-length", type=int, help="override maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="use a random subset of the cleaned data")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset name or local .jsonl file")
    parser.add_argument("--model-name", default=MODEL_NAME, help="base model")
    parser.add_argument("--output-root", default="runs", help="folder for all outputs")
    parser.add_argument("--grad-checkpointing", action="store_true", help="save GPU memory (slower)")
    parser.add_argument("--save-merged", action="store_true", help="also save the adapter merged into the base model")
    args = parser.parse_args()

    # ---- final configuration = named config + command-line overrides
    cfg = dict(CONFIGS[args.config])
    for key in ("rank", "alpha", "lr", "epochs", "batch_size", "grad_accum", "max_length"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["target_modules"] = ATTENTION_MODULES
    cfg["lora_dropout"] = 0.05
    run_name = args.name or args.config
    root = Path(args.output_root)
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device, dtype = get_device(), pick_dtype()
    print(f"Run '{run_name}' | device: {device} | dtype: {dtype}")
    if device == "cpu":
        print("WARNING: no GPU found, training on the CPU will be very slow. Use --max-samples 200 --epochs 1 "
              "for a smoke test, or run on a GPU (see the instructions).")
    print("Configuration:", json.dumps(cfg))

    # ---- Step 1: data
    tokenizer = get_tokenizer(args.model_name)
    raw = load_records(args.dataset)
    rows, removed = preprocess(raw)
    report = inspect_data(raw, rows, removed, tokenizer, cfg["max_length"])
    rng = random.Random(SEED)
    rng.shuffle(rows)
    if args.max_samples:
        rows = rows[:args.max_samples]
    n_val = max(1, round(0.1 * len(rows)))               # 90% / 10% split
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"Using {len(rows)} records: {len(train_rows)} for training, {len(val_rows)} for validation")
    # The split is saved so that the evaluation program tests on exactly the same validation questions
    save_jsonl(root / "data" / "train.jsonl", train_rows)
    save_jsonl(root / "data" / "val.jsonl", val_rows)
    (root / "data" / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- Step 2: prompts -> token ids (prompt tokens masked in the labels)
    def encode_all(records):
        encoded = [encode_example(tokenizer, r["question"], r["answer"], cfg["max_length"]) for r in records]
        return Dataset.from_list([e for e in encoded if e is not None])
    train_ds, val_ds = encode_all(train_rows), encode_all(val_rows)
    print(f"Tokenized examples: {len(train_ds)} train, {len(val_ds)} validation")
    print("Example prompt:\n" + tokenizer.decode(train_ds[0]["input_ids"]) + "\n")

    # ---- Step 3: model + LoRA
    model = load_base_model(args.model_name, dtype)
    lora = LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    for p in model.parameters():                         # LoRA weights are trained in fp32 for stable updates
        if p.requires_grad:
            p.data = p.data.float()
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    # ---- Step 4: fine-tuning
    use_bf16 = device == "cuda" and dtype == torch.bfloat16
    use_fp16 = device == "cuda" and dtype == torch.float16
    targs = make_training_args(run_dir / "checkpoints", cfg, use_bf16, use_fp16)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator)
    start = time.time()
    result = trainer.train()
    runtime = time.time() - start
    final_eval = trainer.evaluate()

    # ---- Step 5: save adapter + metrics
    model.save_pretrained(run_dir / "adapter")
    tokenizer.save_pretrained(run_dir / "adapter")
    if args.save_merged:
        merged = model.merge_and_unload()
        merged.save_pretrained(run_dir / "merged")
        tokenizer.save_pretrained(run_dir / "merged")

    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)       # nothing useful in there (save_strategy="no")
    history = trainer.state.log_history
    train_losses = [h["loss"] for h in history if "loss" in h]
    metrics = {
        "run_name": run_name, "config": cfg, "n_train": len(train_ds), "n_val": len(val_ds),
        "trainable_parameters": trainable, "total_parameters": total,
        "mean_training_loss": round(result.training_loss, 4),               # average over all steps
        "final_training_loss": round(statistics.mean(train_losses[-5:]), 4) if train_losses else None,
        # one entry per epoch (the final trainer.evaluate() call repeats the last epoch, so keep one per epoch)
        "val_loss_per_epoch": [{"epoch": e, "eval_loss": l} for e, l in
                               sorted({round(h["epoch"], 2): round(h["eval_loss"], 4) for h in history if "eval_loss" in h}.items())],
        "final_val_loss": round(final_eval["eval_loss"], 4),
        "train_minutes": round(runtime / 60, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("\n=== Done ===")
    print(json.dumps(metrics, indent=2))
    print(f"Adapter saved to {run_dir / 'adapter'}")


if __name__ == "__main__":
    main()
