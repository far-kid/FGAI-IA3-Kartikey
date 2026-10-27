"""Generate result tables, plot, samples, and the README conclusion."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import DATA, MODELS, RESULTS, RUNS, read_jsonl


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def build():
    test = read_jsonl(DATA / "test.jsonl")
    train_rows = read_jsonl(DATA / "train.jsonl")
    expected_ids = [row["id"] for row in test]
    training, metrics, predictions = {}, {}, {}
    for name in MODELS:
        training[name] = load(RUNS / name / "train_metrics.json")
        metrics[name] = load(RESULTS / f"{name}_metrics.json")
        predictions[name] = read_jsonl(RESULTS / f"{name}_translations.jsonl")
        if training[name].get("max_train_steps_requested"):
            raise ValueError(f"{name} is a smoke run; retrain without --max-train-steps")
        if training[name]["train_examples"] != len(train_rows):
            raise ValueError(f"{name} training count does not match the current split")
        if [row["id"] for row in predictions[name]] != expected_ids:
            raise ValueError(f"{name} predictions do not match the complete test split")
    rows = []
    for name in MODELS:
        t, m = training[name], metrics[name]
        rows.append([name, f'{m["bleu"]:.2f}', f'{m["chrf"]:.2f}', f'{t["train_seconds"] / 60:.1f}',
                     f'{m["mean_seconds_per_example"]:.3f}', f'{m["total_parameters"] / 1e6:.1f}M',
                     f'{t["trainable_parameters"] / 1e6:.1f}M'])
    comparison = table(["Model", "BLEU", "chrF", "Train min", "Inference sec/text", "Total params", "Trained params"], rows)
    temp_rows = [[name, entry["temperature"], entry["examples"], f'{entry["bleu"]:.2f}', f'{entry["chrf"]:.2f}']
                 for name in ("llama", "gemma") for entry in metrics[name]["temperature_analysis"]]
    temperature = table(["Model", "Temperature", "Texts", "BLEU", "chrF"], temp_rows)
    ranked = sorted(MODELS, key=lambda name: metrics[name]["bleu"], reverse=True)
    winner = ranked[0]
    summary = f"""## Measured results

All four systems translated the same {len(test)} held-out texts. Scores are corpus-level SacreBLEU BLEU and chrF. Main results use greedy decoding and at most 80 new tokens. Training time includes validation and checkpoint saving; inference time includes tokenization and generation. Timing comparisons are meaningful only on the same hardware.

{comparison}

**Best by BLEU on this test set:** {winner} ({metrics[winner]['bleu']:.2f} BLEU, {metrics[winner]['chrf']:.2f} chrF). The next highest BLEU was {ranked[1]} ({metrics[ranked[1]]['bleu']:.2f}). This ranking applies to this subset and these training budgets.

### Sampling temperature

The two decoder-only models were also evaluated on the same first {temp_rows[0][2]} test texts at greedy temperature 0 and sampling temperatures 0.7 and 1.0 (top-p 0.9). These are exploratory runs; the main table uses greedy decoding for all models.

{temperature}

Sampling may change wording or cause omissions or additions. The saved examples support closer review.

### Model comparison and limitations

The basic Transformer was trained from scratch; mT5 was fully fine-tuned; Llama and Gemma used LoRA on query and value projections. Pretraining, tokenization, model size, and adaptation also differ, so these results cannot isolate architecture as the cause of score differences. BLEU and chrF use one reference per source; they can penalize valid alternatives and miss serious meaning errors.

### Responsible and ethical use

Older book text can carry stereotypes, outdated expressions, and uneven cultural representation. A model may add facts, lose negation, misgender a person, or mistranslate sensitive medical, legal, religious, or sexual content. Review the paired examples in `results/audit_samples.md` and the complete translation files for these failures. For real use, evaluate representative domains and subgroups, preserve the source text, flag uncertain or consequential translations, and require human review and correction. These student systems should not be treated as authoritative for consequential decisions.
"""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text(summary, encoding="utf-8")
    audit = ["# Paired translation samples", "", "Inspect the JSONL files for the full test set.", ""]
    for index, row in enumerate(test[:15]):
        audit += [f"## Example {index + 1}: ID {row['id']}", "", f"**English:** {row['source']}", "", f"**Reference French:** {row['target']}", ""]
        for name in MODELS:
            audit += [f"**{name}:** {predictions[name][index]['prediction']}", ""]
    (RESULTS / "audit_samples.md").write_text("\n".join(audit), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("bleu", "chrf")):
        bars = ax.bar(MODELS, [metrics[name][metric] for name in MODELS])
        ax.set_ylabel(metric.upper() if metric == "bleu" else "chrF")
        ax.bar_label(bars, fmt="%.1f")
        ax.set_ylim(0, max(metrics[name][metric] for name in MODELS) * 1.2 + 1)
    fig.suptitle("English to French translation: shared test split")
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close(fig)
    path = Path("README.md")
    content = path.read_text(encoding="utf-8")
    begin, end = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    if begin not in content or end not in content:
        raise ValueError("README result markers are missing")
    path.write_text(content.split(begin)[0] + begin + "\n" + summary + "\n" + end + content.split(end, 1)[1], encoding="utf-8")
    print(f"Report ready. Best BLEU: {winner}. Review results/audit_samples.md before submission.")


if __name__ == "__main__":
    build()
