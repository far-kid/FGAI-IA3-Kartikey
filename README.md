# English-to-French machine translation: four architectures

ICT4403 Foundations of Generative AI, Assignment 3. The objective is to translate the same English test passages into French with a basic encoder-decoder Transformer, mT5, Llama, and Gemma, then compare quality and computation and review responsible-use concerns. The assignment text is in [assn.md](assn.md).

## Dataset

[OPUS Books, English-French](https://huggingface.co/datasets/Helsinki-NLP/opus_books) is a public parallel corpus of aligned book text. Its dataset card describes the books as copyright-free and allows personal, educational, and research use; it restricts commercial use and mass redistribution without permission. This repository stores only the selected IDs, preprocessing manifest, metrics, and generated samples. The source corpus is downloaded by `src.prepare_data`.

The script normalizes whitespace, drops empty or extreme-length pairs, removes repeated English sources, seed-shuffles with seed 42, and writes 8,000 training, 500 validation, and 500 test pairs by default. Every model reads those exact files. Source deduplication prevents an identical English sentence from crossing splits. The manifest records counts and a SHA-256 hash of selected source IDs. The split is at sentence level, so passages from the same book may still cross splits; this is a limitation when interpreting scores.

## Models

| Model | Architecture | Adaptation |
| --- | --- | --- |
| Basic Transformer | 4-layer encoder and 4-layer decoder, 256 hidden units, 8 attention heads | Trained from scratch with a shared 8k SentencePiece vocabulary |
| [mT5-small](https://huggingface.co/google/mt5-small) | Encoder-decoder | Full fine-tuning |
| [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Decoder-only | LoRA on query and value attention projections |
| [Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it) | Decoder-only | LoRA on query and value attention projections |

The Llama and Gemma model pages may require signing in and accepting model terms before their weights can be downloaded. A Hugging Face token with read access is needed in the cloud session. Do not commit tokens or model weights.

## Run on a cloud GPU

Use a Linux GPU runtime with Python 3.10-3.12 and enough storage for the model downloads. The commands below assume a fresh runtime and that this repository has been cloned. A 16 GB T4 is the intended starting point; if the service gives a smaller GPU, reduce batch size in `scripts/run_cloud.sh` and record the change.

```bash
python -m pip install -r requirements.txt
hf auth login
bash scripts/run_cloud.sh
```

Before running, open the [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [Gemma](https://huggingface.co/google/gemma-3-270m-it) pages while signed in and accept any access conditions. `hf auth login` prompts for your personal read token. If the `hf` executable is not on PATH, run `python -m huggingface_hub.commands.huggingface_cli login`.

The runner prepares data, trains one model at a time, evaluates all four on the full test set, and builds the report. For a small pipeline check before the full run, prepare the data and use `python -m src.train --model transformer --epochs 1 --max-train-steps 2`; do not submit smoke-run metrics. Each stage can also be run separately; see [run_cloud.sh](scripts/run_cloud.sh). Keep the same `data/` folder between stages.

Generated artifacts:

- `data/manifest.json`: dataset details and split hash.
- `runs/<model>/train_metrics.json`: elapsed training time, validation loss, and parameter counts.
- `results/<model>_metrics.json`: corpus BLEU, chrF, inference time, parameter count, and decoder temperature study.
- `results/<model>_translations.jsonl`: full paired test predictions.
- `results/audit_samples.md`, `results/model_comparison.png`, and `results/summary.md`: qualitative samples, plot, and measured comparison.

`data/*.jsonl` and `runs/` are ignored because they contain downloaded text and large checkpoints. Keep them in persistent cloud storage until evaluation is finished. Commit `data/manifest.json`, `results/`, and the updated `README.md` after reviewing the outputs. Do not commit Hugging Face tokens.

## Evaluation and results

SacreBLEU corpus BLEU and chrF score the same held-out references. The principal comparison uses greedy decoding and a maximum of 80 new tokens for all models. Training time, mean inference seconds per text, and actual model parameter counts are measured during the run. Decoder-only models are additionally sampled at temperatures 0.7 and 1.0 on the same 100-item test subset; temperature 0 is the greedy baseline. The report includes architectural caveats and responsible-use analysis.

<!-- RESULTS:BEGIN -->
**Results pending cloud GPU execution.** No model has been trained or scored in this repository yet, so there is no defensible best-performing model to name. Running `python -m src.report` after all four evaluations replaces this section with the measured table, temperature analysis, comparison, and conclusion.
<!-- RESULTS:END -->

## Reproducibility notes

The random seed is 42. Training batches, epoch counts, and learning rates are specified in [run_cloud.sh](scripts/run_cloud.sh); exact package versions and hardware are recorded with the final metrics. Score comparisons remain affected by pretraining data, model size, tokenizer, and adaptation method. Machine translation of sensitive text requires human review, even when automated scores are high.
