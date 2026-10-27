"""Score every model on the same held-out rows using greedy decoding."""

import argparse
import json
import time

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from .basic_transformer import BasicTransformer, load_sentencepiece
from .common import DATA, HF_MODELS, MODELS, PROMPT, RESULTS, RUNS, env_info, read_jsonl, seed_everything, sync_cuda, write_json, write_jsonl


def load_model(name: str):
    run_dir = RUNS / name
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    if name == "transformer":
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        model = BasicTransformer(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer = load_sentencepiece(run_dir / "spm.model")
    else:
        tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        dtype = torch.float16 if gpu else torch.float32
        if name == "mt5":
            model = AutoModelForSeq2SeqLM.from_pretrained(run_dir / "model", torch_dtype=dtype)
        else:
            base = AutoModelForCausalLM.from_pretrained(HF_MODELS[name], torch_dtype=dtype)
            model = PeftModel.from_pretrained(base, run_dir / "model")
    model.to(device)
    model.eval()
    return model, tokenizer


def normalize_prediction(text: str) -> str:
    """Apply the same minimal cleanup to all generated text."""
    text = text.strip()
    if text.lower().startswith("french:"):
        text = text[7:].strip()
    return text.splitlines()[0].strip() if text else ""


@torch.inference_mode()
def predict(name, model, tokenizer, source, *, temperature=0.0, max_new_tokens=80):
    device = next(model.parameters()).device
    if name == "transformer":
        ids = torch.tensor([tokenizer.encode(source)[:94] + [3]], device=device)
        output = model.translate(ids, max_new_tokens=max_new_tokens)[0].tolist()
        output = [item for item in output[1:] if item not in (0, 3)]
        text = tokenizer.decode(output)
    elif name == "mt5":
        encoded = tokenizer("translate English to French: " + source, return_tensors="pt", truncation=True, max_length=128).to(device)
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                   **({"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}))
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
    else:
        encoded = tokenizer(PROMPT.format(source=source), return_tensors="pt", truncation=True, max_length=112).to(device)
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                   pad_token_id=tokenizer.eos_token_id,
                                   **({"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}))
        text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    return normalize_prediction(text)


def score(predictions, references):
    return {
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf": sacrebleu.corpus_chrf(predictions, [references]).score,
    }


def evaluate(name: str, *, limit: int | None = None, temperature_samples: int = 100):
    rows = read_jsonl(DATA / "test.jsonl")
    if limit is not None:
        rows = rows[:limit]
    model, tokenizer = load_model(name)
    seed_everything(42)
    predictions = []
    sync_cuda()
    start = time.perf_counter()
    for i, row in enumerate(rows):
        predictions.append(predict(name, model, tokenizer, row["source"]))
        if (i + 1) % 50 == 0:
            print(f"{name}: {i + 1}/{len(rows)}", flush=True)
    sync_cuda()
    inference_seconds = time.perf_counter() - start
    references = [row["target"] for row in rows]
    metrics = {
        "model": name,
        "decoding": "greedy",
        "max_new_tokens": 80,
        "test_examples": len(rows),
        "inference_seconds": inference_seconds,
        "mean_seconds_per_example": inference_seconds / len(rows),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        **score(predictions, references), **env_info(),
    }
    write_jsonl(RESULTS / f"{name}_translations.jsonl", [
        {"id": row["id"], "source": row["source"], "reference": row["target"], "prediction": prediction}
        for row, prediction in zip(rows, predictions)
    ])
    if name in ("llama", "gemma") and temperature_samples > 0:
        subset = rows[:min(temperature_samples, len(rows))]
        temperature_results = []
        for temperature in (0.0, 0.7, 1.0):
            seed_everything(42)
            samples = [predict(name, model, tokenizer, row["source"], temperature=temperature) for row in subset]
            temp_score = score(samples, [row["target"] for row in subset])
            temperature_results.append({"temperature": temperature, "examples": len(subset), **temp_score})
            write_jsonl(RESULTS / f"{name}_temperature_{str(temperature).replace('.', '_')}.jsonl", [
                {"id": row["id"], "source": row["source"], "reference": row["target"], "prediction": sample}
                for row, sample in zip(subset, samples)
            ])
        metrics["temperature_analysis"] = temperature_results
    write_json(RESULTS / f"{name}_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for reported results")
    parser.add_argument("--temperature-samples", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model, limit=args.limit, temperature_samples=args.temperature_samples)


if __name__ == "__main__":
    main()
