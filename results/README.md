# Results exported from the executed notebook

These files were recovered from the saved outputs of [`ia3_machine_translation.ipynb`](../notebooks/ia3_machine_translation.ipynb). The notebook reports a full run on a Tesla T4 with `QUICK_RUN = False`, seed 42, and 500 sampled captions from the official Multi30k 2016 Flickr test split.

- `model_metrics.csv` records the six evaluated systems, BLEU, chrF, time, and parameters.
- The PNG files are the plots saved in the notebook outputs.
- The other CSV files contain the displayed tokenization, length, error, gender, and sensitive-content tables.
- `sample_translations.txt` and `worst_translations.txt` contain examples printed by the notebook.

The complete per-sentence translation CSVs and probe CSVs were written to `/kaggle/working/mt_outputs` during the original run but are not embedded in the notebook. They should be added from the notebook's `mt_outputs.zip` archive when available. Do not interpret these exported tables as a fresh rerun or as independent verification of the GPU results.

Multi30k originates from the [official dataset repository](https://github.com/multi30k/dataset) and was described in the [Multi30K paper](https://aclanthology.org/W16-3210/). The paper states a CC BY-NC-SA 4.0 license for the dataset; this repository uses small excerpts for noncommercial coursework with attribution.
