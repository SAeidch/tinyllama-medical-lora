"""
evaluate_lora.py - Problem 3, program 2: test the fine-tuned models

For the un-finetuned TinyLlama ("base") and for every fine-tuned configuration it computes:
  * training loss (read from the training run) and validation loss (recomputed here, so that the
    base model and all configurations are measured in exactly the same way)
  * medical question-answer accuracy, exact match and token F1
  * ROUGE-L and BERTScore against the reference answers
  * hallucination / factual consistency (NLI-based, plus a simple word-overlap measure)
  * generalization: the same generation metrics on questions that are NOT in the training data
    (a hand-written set of medical questions, and optionally questions from another Hugging Face dataset)

Usage (after training the configurations with train_lora.py):
    python evaluate_lora.py --configs baseline rank16 highlr
Faster, lighter options:  --bertscore-model distilbert-base-uncased   --max-eval-samples 100
                          --no-bertscore   --no-nli
Results are written to runs/eval/ (results.md, results.json, predictions_*.jsonl, examples.md).

How the answer-quality metrics are defined (all are automatic proxies, not human judgment):
  accuracy              share of questions where the generated answer contains the key words of the reference
                        answer: all of them if the reference has at most 2 content words (e.g. "Vitamin C"),
                        otherwise at least 50% of them (stop words removed)
  exact match / F1      normalized string equality / word-overlap F1 with the reference answer
  ROUGE-L, BERTScore    standard similarity scores against the reference answer (F-measure)
  factual consistency   an NLI model checks the generated answer against the reference answer in both
                        directions; an answer is "consistent" if either direction is entailment
  hallucination rate    share of answers that are NOT consistent with the reference (neutral or contradicting)
  contradiction rate    share of answers that contradict the reference
  unsupported words     share of content words in the answer found neither in the question nor in the reference
"""

import argparse
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lora_common import (DATASET_NAME, MODEL_NAME, SEED, build_prompt, encode_example, get_device, get_tokenizer,
                         load_base_model, load_jsonl, save_jsonl)

# --------------------------------------------------------------------------
# Test questions outside the training data
# --------------------------------------------------------------------------
# 30 short, textbook medical facts written by hand (not taken from the training dataset)
OOD_BUILTIN = [
    ("What is the first-line drug for anaphylaxis?", "Intramuscular epinephrine (adrenaline)."),
    ("Deficiency of which vitamin causes scurvy?", "Vitamin C (ascorbic acid)."),
    ("What is the most common cause of community-acquired pneumonia?", "Streptococcus pneumoniae."),
    ("Which electrolyte abnormality is classically associated with peaked T waves on an ECG?", "Hyperkalemia."),
    ("What is the antidote for acetaminophen (paracetamol) overdose?", "N-acetylcysteine."),
    ("Which hormone is deficient in type 1 diabetes mellitus?", "Insulin."),
    ("Which bacterium causes most peptic ulcers and is treated with triple therapy?", "Helicobacter pylori."),
    ("Which organ produces bile?", "The liver."),
    ("What is the most common type of thyroid cancer?", "Papillary thyroid carcinoma."),
    ("Which anticoagulant is monitored with the INR?", "Warfarin."),
    ("Which antibody class is produced first in a primary immune response?", "IgM."),
    ("Deficiency of which vitamin causes Wernicke encephalopathy?", "Thiamine (vitamin B1)."),
    ("What is the most common cause of bacterial meningitis in adults?", "Streptococcus pneumoniae."),
    ("Which nerve is compressed in carpal tunnel syndrome?", "The median nerve."),
    ("What is the first-line treatment for status epilepticus?", "Benzodiazepines."),
    ("What is the antidote for heparin overdose?", "Protamine sulfate."),
    ("What is the antidote for opioid overdose?", "Naloxone."),
    ("Which cranial nerve is affected in Bell palsy?", "The facial nerve."),
    ("What is the most common cause of hypothyroidism in iodine-sufficient regions?", "Hashimoto thyroiditis."),
    ("Which part of the nephron reabsorbs most of the filtered glucose?", "The proximal convoluted tubule."),
    ("Which organism causes syphilis?", "Treponema pallidum."),
    ("Deficiency of which vitamin causes night blindness?", "Vitamin A."),
    ("Which chromosomal abnormality causes Down syndrome?", "Trisomy 21."),
    ("What is the drug of choice for Pneumocystis jirovecii pneumonia?", "Trimethoprim-sulfamethoxazole."),
    ("Which type of hypersensitivity reaction is anaphylaxis?", "Type I hypersensitivity."),
    ("Which enzyme is deficient in phenylketonuria?", "Phenylalanine hydroxylase."),
    ("What is the first-line treatment for an acute asthma exacerbation?", "Inhaled short-acting beta-2 agonists such as albuterol."),
    ("Which organism is the most common cause of urinary tract infections?", "Escherichia coli."),
    ("What is the antidote for benzodiazepine overdose?", "Flumazenil."),
    ("Which vitamin is given to prevent neural tube defects in pregnancy?", "Folic acid (folate)."),
]
# Non-medical prompts, only used to look for side effects of fine-tuning on general behaviour (no scores)
GENERAL_PROMPTS = ["What is the capital of France?", "Write one sentence greeting a new colleague.",
                   "What is 12 times 8?", "Explain what a neural network is in one sentence.",
                   "Give me one tip for studying effectively."]

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those which who whom what when where why how do does did has have had will would can could should may might must
into than then there their they them he she his her you your we our i not no yes also such very more most some any other""".split()) - {"no", "not", "yes"}


# --------------------------------------------------------------------------
# Simple text metrics (no model needed)
# --------------------------------------------------------------------------
def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def content_words(text):
    return {t for t in normalize(text).split() if t not in STOPWORDS}   # single letters are kept: vitamin C is not vitamin D


def token_f1(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    common = sum((min(p.count(w), r.count(w)) for w in set(p) & set(r)))
    if not p or not r or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def is_correct(pred, ref, fraction=0.5):
    """
    Automatic accuracy rule: the generated answer is counted as correct if it contains the key words
    of the reference answer. Reference answers with at most 2 content words (e.g. "Vitamin C.") must
    be matched completely; longer references need at least `fraction` of their content words.
    """
    ref_words = content_words(ref)
    if not ref_words:                                     # reference made only of stop words, e.g. "No."
        return bool(normalize(ref)) and normalize(ref) in normalize(pred)
    overlap = len(ref_words & content_words(pred))
    required = len(ref_words) if len(ref_words) <= 2 else math.ceil(fraction * len(ref_words))
    return overlap >= required


def unsupported_rate(pred, ref, question):
    """Share of content words in the prediction that appear neither in the question nor in the reference."""
    pred_words = content_words(pred)
    if not pred_words:
        return 0.0
    return len(pred_words - content_words(ref) - content_words(question)) / len(pred_words)


# --------------------------------------------------------------------------
# Model-based scorers (loaded once, only when needed)
# --------------------------------------------------------------------------
def make_rouge():
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return lambda preds, refs: [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def make_bertscore(model_type, device):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=model_type, lang="en", device=device, batch_size=16)
    def run(preds, refs):
        _, _, f1 = scorer.score([p or " " for p in preds], [r or " " for r in refs])
        return [float(x) for x in f1]
    return run


class NLIScorer:
    """Natural-language-inference model: probabilities of entailment / neutral / contradiction."""

    def __init__(self, model_name, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.index = {}
        for i, label in self.model.config.id2label.items():       # label order differs between NLI models
            for name in ("entailment", "neutral", "contradiction"):
                if name in label.lower():
                    self.index[name] = int(i)
        assert len(self.index) == 3, f"unexpected NLI labels: {self.model.config.id2label}"

    @torch.no_grad()
    def predict(self, premises, hypotheses, batch_size=16):
        out = []
        for i in range(0, len(premises), batch_size):
            enc = self.tokenizer(premises[i:i + batch_size], hypotheses[i:i + batch_size], truncation=True,
                                 max_length=512, padding=True, return_tensors="pt").to(self.device)
            probs = F.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            out += [{name: float(p[idx]) for name, idx in self.index.items()} for p in probs]
        return out


# --------------------------------------------------------------------------
# Aggregate metrics for one set of (question, reference, prediction) rows
# --------------------------------------------------------------------------
def score_predictions(rows, rouge, bertscore, nli, threshold=0.5):
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]
    qs = [r["question"] for r in rows]
    mean = lambda xs: round(float(statistics.mean(xs)), 4)
    m = {"n": len(rows),
         "accuracy": mean([is_correct(p, r, threshold) for p, r in zip(preds, refs)]),
         "exact_match": mean([normalize(p) == normalize(r) for p, r in zip(preds, refs)]),
         "token_f1": mean([token_f1(p, r) for p, r in zip(preds, refs)]),
         "unsupported_word_rate": mean([unsupported_rate(p, r, q) for p, r, q in zip(preds, refs, qs)]),
         "avg_answer_words": round(float(statistics.mean(len(p.split()) for p in preds)), 1)}
    if rouge:
        m["rougeL"] = mean(rouge(preds, refs))
    if bertscore:
        m["bertscore_f1"] = mean(bertscore(preds, refs))
    if nli:
        hyp = [p or "(no answer)" for p in preds]
        fwd = nli.predict(refs, hyp)                       # reference -> answer
        bwd = nli.predict(hyp, refs)                       # answer -> reference
        pick = lambda d: max(d, key=d.get)
        consistent = [pick(f) == "entailment" or pick(b) == "entailment" for f, b in zip(fwd, bwd)]
        contradicted = [(not c) and "contradiction" in (pick(f), pick(b)) for c, f, b in zip(consistent, fwd, bwd)]
        m["factual_consistency_score"] = mean([max(f["entailment"], b["entailment"]) for f, b in zip(fwd, bwd)])
        m["consistent_rate"] = mean(consistent)
        m["hallucination_rate"] = round(1 - m["consistent_rate"], 4)
        m["contradiction_rate"] = mean(contradicted)
    return m


# --------------------------------------------------------------------------
# Generation and validation loss
# --------------------------------------------------------------------------
@torch.no_grad()
def generate_answers(model, tokenizer, questions, batch_size, max_new_tokens, device):
    """Greedy decoding (deterministic), identical settings for every model."""
    tokenizer.padding_side = "left"
    answers = []
    for i in range(0, len(questions), batch_size):
        prompts = [build_prompt(tokenizer, q) for q in questions[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        answers += [t.strip() for t in tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return answers


@torch.no_grad()
def compute_val_loss(model, tokenizer, rows, max_length, batch_size, device):
    """Average loss per ANSWER token over the validation rows (token-weighted), and the perplexity."""
    examples = [e for e in (encode_example(tokenizer, r["question"], r["reference"], max_length) for r in rows) if e]
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        width = max(len(e["input_ids"]) for e in batch)
        ids = torch.tensor([e["input_ids"] + [tokenizer.pad_token_id] * (width - len(e["input_ids"])) for e in batch])
        mask = torch.tensor([e["attention_mask"] + [0] * (width - len(e["attention_mask"])) for e in batch])
        labels = torch.tensor([e["labels"] + [-100] * (width - len(e["labels"])) for e in batch])
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, :-1].float()
        target = labels[:, 1:].to(device)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100,
                                 reduction="sum").item()
        count += int((target != -100).sum())
    loss = total / max(count, 1)
    return round(loss, 4), round(math.exp(min(loss, 20)), 2)


# --------------------------------------------------------------------------
# Data for the evaluation sets
# --------------------------------------------------------------------------
def load_external_ood(name, n):
    """Questions from another Hugging Face dataset (different source than the training flashcards)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split="train")
        rows = [{"question": " ".join((r.get("input") or "").split()), "reference": " ".join((r.get("output") or "").split())}
                for r in ds]
        rows = [r for r in rows if r["question"] and 0 < len(r["reference"]) <= 600 and len(r["question"]) <= 300]
        random.Random(SEED).shuffle(rows)
        return rows[:n]
    except Exception as err:
        print(f"WARNING: could not load the external out-of-distribution set '{name}': {err}")
        return []


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------
def fmt(v, key=None):
    if v is None:
        return "-"
    if key == "avg_answer_words":
        return f"{v:.1f}"
    return f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else str(v)


def markdown_table(title, variants, results, set_name, keys):
    header = ["Model"] + [k[1] for k in keys]
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for v in variants:
        m = results[v].get(set_name)
        if m:
            lines.append("| " + " | ".join([v] + [fmt(m.get(k[0]), k[0]) for k in keys]) + " |")
    return "\n".join(lines) + "\n"


def write_reports(out_dir, variants, results, losses, train_metrics, preds_by_variant, n_examples=5):
    gen_keys = [("accuracy", "Accuracy"), ("exact_match", "Exact match"), ("token_f1", "Token F1"), ("rougeL", "ROUGE-L"),
                ("bertscore_f1", "BERTScore F1"), ("factual_consistency_score", "Factual consistency"),
                ("hallucination_rate", "Hallucination rate"), ("contradiction_rate", "Contradiction rate"),
                ("unsupported_word_rate", "Unsupported words"), ("avg_answer_words", "Avg. words")]
    md = ["# Evaluation results", "", "**Losses**", "",
          "| Model | Mean training loss | Final training loss | Validation loss | Validation perplexity |", "|---|---|---|---|---|"]
    for v in variants:
        t = train_metrics.get(v, {})
        md.append(f"| {v} | {fmt(t.get('mean_training_loss'))} | {fmt(t.get('final_training_loss'))} | "
                  f"{fmt(losses[v][0])} | {fmt(losses[v][1])} |")
    md.append("")
    md.append(markdown_table("Validation questions (from the training dataset, not used in training)", variants, results, "val", gen_keys))
    md.append(markdown_table("Out-of-dataset questions: hand-written medical questions", variants, results, "ood_builtin", gen_keys))
    if any("ood_external" in results[v] for v in variants):
        md.append(markdown_table("Out-of-dataset questions: other Hugging Face dataset", variants, results, "ood_external", gen_keys))
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps({"losses": {v: {"val_loss": losses[v][0], "val_perplexity": losses[v][1]} for v in variants},
                                                      "training": train_metrics, "generation_metrics": results}, indent=2), encoding="utf-8")

    ex = ["# Example answers", ""]
    for set_name, title in (("val", "Validation questions"), ("ood_builtin", "Hand-written out-of-dataset questions"),
                            ("general", "General (non-medical) prompts")):
        first = preds_by_variant[variants[0]].get(set_name, [])[:n_examples]
        ex += [f"## {title}", ""]
        for i, row in enumerate(first):
            ex.append(f"**Question:** {row['question']}  ")
            if row["reference"]:
                ex.append(f"**Reference:** {row['reference']}  ")
            for v in variants:
                ex.append(f"**{v}:** {preds_by_variant[v][set_name][i]['prediction']}  ")
            ex.append("")
    (out_dir / "examples.md").write_text("\n".join(ex), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the base TinyLlama and the fine-tuned LoRA configurations.")
    parser.add_argument("--configs", nargs="+", default=["baseline", "rank16", "highlr"], help="run names to evaluate")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-eval-samples", type=int, default=200, help="validation questions to generate answers for")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512, help="max sequence length for the validation loss")
    parser.add_argument("--accuracy-threshold", type=float, default=0.5, help="share of key words needed when the reference has 3 or more")
    parser.add_argument("--bertscore-model", default="roberta-large", help="e.g. distilbert-base-uncased for speed")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--ood-dataset", default="medalpaca/medical_meadow_wikidoc", help="external out-of-dataset questions")
    parser.add_argument("--ood-samples", type=int, default=50)
    parser.add_argument("--no-ood-external", action="store_true")
    parser.add_argument("--skip-generation", action="store_true", help="recompute metrics from saved predictions")
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    out_dir = runs / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    variants = ["base"] + [c for c in args.configs if (runs / c / "adapter").exists()]
    missing = [c for c in args.configs if not (runs / c / "adapter").exists()]
    if missing:
        print(f"WARNING: no trained adapter found for {missing}; they are skipped.")
    print("Models to evaluate:", variants)

    # ---- evaluation sets (all with question / reference)
    val_rows = [{"question": r["question"], "reference": r["answer"]} for r in load_jsonl(runs / "data" / "val.jsonl")]
    sets = {"val": val_rows[:args.max_eval_samples],
            "ood_builtin": [{"question": q, "reference": a} for q, a in OOD_BUILTIN]}
    if not args.no_ood_external:
        ext = load_external_ood(args.ood_dataset, args.ood_samples)
        if ext:
            sets["ood_external"] = ext
    general = [{"question": q, "reference": ""} for q in GENERAL_PROMPTS]
    print({k: len(v) for k, v in sets.items()})

    # ---- phase 1: validation loss and generation for every model
    losses, preds_by_variant = {}, {}
    if args.skip_generation:
        for v in variants:
            preds_by_variant[v] = {}
            for row in load_jsonl(out_dir / f"predictions_{v}.jsonl"):
                preds_by_variant[v].setdefault(row["set"], []).append(row)
            losses[v] = tuple(json.loads((out_dir / f"loss_{v}.json").read_text())["val"])
    else:
        from peft import PeftModel
        tokenizer = get_tokenizer(args.model_name)
        model = load_base_model(args.model_name).eval()
        peft_model = None
        for v in variants:
            if v != "base":                                   # attach the LoRA adapter of this configuration
                adapter = str(runs / v / "adapter")
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(model, adapter, adapter_name=v)
                else:
                    peft_model.load_adapter(adapter, adapter_name=v)
                peft_model.set_adapter(v)
                model = peft_model
                model.eval()
            t0 = time.time()
            losses[v] = compute_val_loss(model, tokenizer, sets["val"], args.max_length, 4, device)
            (out_dir / f"loss_{v}.json").write_text(json.dumps({"val": losses[v]}))
            preds_by_variant[v] = {}
            rows_out = []
            for set_name, rows in {**sets, "general": general}.items():
                answers = generate_answers(model, tokenizer, [r["question"] for r in rows], args.batch_size,
                                           args.max_new_tokens, device)
                preds_by_variant[v][set_name] = [{"set": set_name, "question": r["question"], "reference": r["reference"],
                                                  "prediction": a} for r, a in zip(rows, answers)]
                rows_out += preds_by_variant[v][set_name]
            save_jsonl(out_dir / f"predictions_{v}.jsonl", rows_out)
            print(f"[{v}] val loss {losses[v][0]} (ppl {losses[v][1]}), generation done in {time.time() - t0:.0f}s")
        del model, peft_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- phase 2: scores computed from the saved predictions
    rouge = make_rouge()
    bertscore = None if args.no_bertscore else make_bertscore(args.bertscore_model, device)
    nli = None if args.no_nli else NLIScorer(args.nli_model, device)
    results = {}
    for v in variants:
        results[v] = {}
        for set_name in sets:
            results[v][set_name] = score_predictions(preds_by_variant[v][set_name], rouge, bertscore, nli, args.accuracy_threshold)
        print(f"[{v}] validation accuracy {results[v]['val']['accuracy']}, ROUGE-L {results[v]['val'].get('rougeL')}")

    train_metrics = {}
    for v in variants:
        p = runs / v / "metrics.json"
        if v != "base" and p.exists():
            train_metrics[v] = json.loads(p.read_text())
    write_reports(out_dir, variants, results, losses, train_metrics, preds_by_variant)
    print(f"\nResults written to {out_dir}/ (results.md, results.json, examples.md, predictions_*.jsonl)")
    print((out_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
