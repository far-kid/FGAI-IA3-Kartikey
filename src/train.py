"""Train the four assignment models on identical prepared splits."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from .basic_transformer import BasicTransformer, load_sentencepiece, train_sentencepiece
from .common import DATA, HF_MODELS, PROMPT, RUNS, env_info, read_jsonl, seed_everything, sync_cuda, write_json


def transformer_examples(rows, sp):
    return [
        (
            torch.tensor((sp.encode(r["source"])[:94] + [3]), dtype=torch.long),
            torch.tensor(([2] + sp.encode(r["target"])[:94] + [3]), dtype=torch.long),
        )
        for r in rows
    ]


def transformer_collate(batch):
    source = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=0)
    target = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
    return source, target


def seq2seq_examples(rows, tokenizer):
    inputs = tokenizer(["translate English to French: " + r["source"] for r in rows], truncation=True, max_length=128)
    targets = tokenizer(text_target=[r["target"] for r in rows], truncation=True, max_length=96)
    return [(torch.tensor(a), torch.tensor(b)) for a, b in zip(inputs["input_ids"], targets["input_ids"])]


def causal_examples(rows, tokenizer):
    examples = []
    for row in rows:
        prefix = tokenizer(PROMPT.format(source=row["source"]), truncation=True, max_length=112)["input_ids"]
        answer = tokenizer(" " + row["target"], add_special_tokens=False, truncation=True, max_length=79)["input_ids"]
        ids = prefix + answer + [tokenizer.eos_token_id]
        labels = [-100] * len(prefix) + answer + [tokenizer.eos_token_id]
        examples.append((torch.tensor(ids), torch.tensor(labels)))
    return examples


def language_model_collate(batch, pad_id):
    ids = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(x[0]) for x in batch])
    attention_mask = (torch.arange(ids.size(1))[None, :] < lengths[:, None]).long()
    return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}


def optimize(model, train_loader, val_loader, *, epochs, lr, grad_accum, max_train_steps, run_dir, basic=False):
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=gpu)
    best = float("inf")
    best_epoch = 0
    history = []
    steps = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index, batch in enumerate(train_loader):
            if basic:
                source, target = [x.to(device) for x in batch]
                with torch.autocast("cuda", enabled=gpu):
                    logits = model(source, target[:, :-1])
                    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1), ignore_index=0)
            else:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", enabled=gpu):
                    loss = model(**batch).loss
            total_loss += loss.item()
            scaler.scale(loss / grad_accum).backward()
            if (index + 1) % grad_accum == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if max_train_steps and steps >= max_train_steps:
                    break
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if basic:
                    source, target = [x.to(device) for x in batch]
                    with torch.autocast("cuda", enabled=gpu):
                        logits = model(source, target[:, :-1])
                        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1), ignore_index=0)
                else:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast("cuda", enabled=gpu):
                        loss = model(**batch).loss
                val_total += loss.item()
        val_loss = val_total / len(val_loader)
        record = {"epoch": epoch, "train_loss": total_loss / (index + 1), "validation_loss": val_loss, "optimizer_steps": steps}
        history.append(record)
        print(record, flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            if basic:
                torch.save({"config": model.config, "state_dict": model.state_dict()}, run_dir / "model.pt")
            else:
                model.save_pretrained(run_dir / "model")
        if max_train_steps and steps >= max_train_steps:
            break
    sync_cuda()
    return {"train_seconds": time.perf_counter() - start, "best_validation_loss": best, "best_epoch": best_epoch, "history": history, "optimizer_steps": steps}


def train(name: str, args) -> None:
    seed_everything(args.seed)
    train_rows, val_rows = read_jsonl(DATA / "train.jsonl"), read_jsonl(DATA / "validation.jsonl")
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    if name == "transformer":
        sp_path = train_sentencepiece(train_rows, run_dir, args.vocab_size)
        sp = load_sentencepiece(sp_path)
        model = BasicTransformer(sp.vocab_size()).to("cuda" if gpu else "cpu")
        train_examples = transformer_examples(train_rows, sp)
        val_examples = transformer_examples(val_rows, sp)
        collate = transformer_collate
        lr = args.lr or 0.0003
    else:
        model_id = HF_MODELS[name]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if name == "mt5":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16 if gpu else torch.float32)
            train_examples = seq2seq_examples(train_rows, tokenizer)
            val_examples = seq2seq_examples(val_rows, tokenizer)
            collate = lambda batch: language_model_collate(batch, tokenizer.pad_token_id)
            lr = args.lr or 0.0001
        else:
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if gpu else torch.float32)
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"])
            model = get_peft_model(model, lora)
            train_examples = causal_examples(train_rows, tokenizer)
            val_examples = causal_examples(val_rows, tokenizer)
            collate = lambda batch: language_model_collate(batch, tokenizer.pad_token_id)
            lr = args.lr or 0.0002
        model.to("cuda" if gpu else "cpu")
        tokenizer.save_pretrained(run_dir / "tokenizer")
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_examples, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    result = optimize(model, train_loader, val_loader, epochs=args.epochs, lr=lr, grad_accum=args.grad_accum,
                      max_train_steps=args.max_train_steps, run_dir=run_dir, basic=name == "transformer")
    result.update({"model": name, "base_model": HF_MODELS.get(name, "scratch"), "total_parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "train_examples": len(train_rows), "validation_examples": len(val_rows),
                   "batch_size": args.batch_size, "gradient_accumulation": args.grad_accum, "epochs_requested": args.epochs,
                   "seed": args.seed, "learning_rate": lr, **env_info()})
    write_json(run_dir / "train_metrics.json", result)
    print(f"Saved {run_dir / 'train_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["transformer", "mt5", "llama", "gemma"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, help="Only for a quick smoke run; omit for reported results")
    args = parser.parse_args()
    train(args.model, args)


if __name__ == "__main__":
    main()
