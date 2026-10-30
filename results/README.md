# Measured results

These files come from the completed Kaggle run of [the notebook](../notebooks/ia3_machine_translation.ipynb). The run used a Tesla T4, seed 42, the cleaned Multi30k English-German training data, and the same 500-item sample of the official test split for all six systems.

- [`results.json`](results.json) and [`results_table.csv`](results_table.csv) contain BLEU, chrF, elapsed training and inference times, parameter counts, GPU memory, and prompt length.
- `translations_*.csv` contain every test source, reference, and model hypothesis. The same ordered source/reference rows occur in all six files.
- `probes_*.csv` contain the gender, culture, idiom, sensitive-content, and out-of-domain probe translations.
- `*_history.csv` contain training loss logs; the PNG files show the displayed plots.
- [`split_test.csv`](split_test.csv) records the selected test sample. The full Multi30k corpus is available from the [official source](https://github.com/multi30k/dataset).

The reported BLEU and chrF values were independently recomputed from all six translation CSVs and matched `results.json`. The complete model checkpoints and training-data copies are omitted because the notebook downloads or trains them and they are large. The Multi30k paper states a [CC BY-NC-SA 4.0 license](https://aclanthology.org/W16-3210/) for the dataset; these excerpts are used for noncommercial coursework with attribution.
