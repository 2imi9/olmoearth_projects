# Fine-tuning OLMo on the US Power & Climate datasets (tutorial)

This tutorial lives under `docs/tutorials/` so you can `cd docs/tutorials` and
open it directly. It walks through fine-tuning an OLMo sequence-classification
checkpoint on the three public Hugging Face datasets maintained by `2imi9`:

- `2imi9/us-power-climate-data-streamflow`
- `2imi9/us-power-climate-data-climate`
- `2imi9/us-power-climate-data-plants`

The helper script at `scripts/tools/hf_finetune_power_risk.py` merges those
datasets, attaches the right head for regression or classification, and can
push your checkpoint back to the Hub.

## 1) Environment setup
Install the Hugging Face stack (not included in the base `pyproject.toml`) and
authenticate with the Hub so the script can download data and push artifacts:
```bash
uv add transformers datasets evaluate numpy
huggingface-cli login
```

## 2) Pick your target and text columns
Decide which field in the datasets should be tokenized (pass via
`--text-column`) and which field is your label (`--label-column`). If labels are
floats, keep the default `--task-type regression`; if they are integer class
IDs, use `--task-type classification`.

## 3) Run training (regression example)
Fine-tune OLMo-2B-LM on the merged datasets for a numeric target:
```bash
uv run python scripts/tools/hf_finetune_power_risk.py \
  --model-name allenai/OLMo-2B-LM \
  --text-column description \
  --label-column target_value \
  --task-type regression \
  --datasets \
    2imi9/us-power-climate-data-streamflow \
    2imi9/us-power-climate-data-climate \
    2imi9/us-power-climate-data-plants \
  --num-train-epochs 5 \
  --learning-rate 3e-5 \
  --batch-size 8 \
  --push-to-hub-id YOUR_USERNAME/power-risk-olmo \
  --hub-private-repo
```

## 4) Run training (classification example)
If your label column contains integer class IDs, swap `--task-type` and point to
the class column:
```bash
uv run python scripts/tools/hf_finetune_power_risk.py \
  --model-name allenai/OLMo-2B-LM \
  --text-column description \
  --label-column risk_bucket \
  --task-type classification \
  --datasets \
    2imi9/us-power-climate-data-streamflow \
    2imi9/us-power-climate-data-climate \
    2imi9/us-power-climate-data-plants \
  --push-to-hub-id YOUR_USERNAME/power-risk-olmo-classifier
```

## 5) What the script does for you
- Downloads each dataset above and casts the label column to the right type for
  regression or classification.
- Concatenates the training splits and uses validation/test splits when
  available. If none exist, it automatically carves out a validation set from
  the merged training data.
- Tokenizes the chosen text column with a `transformers` tokenizer, sets up an
  appropriate head, and trains with the 🤗 `Trainer` API.
- Saves the fine-tuned model locally, or pushes it to the Hugging Face Hub when
  `--push-to-hub-id` is provided.

## 6) Customize for your energy risk task
- **Use different checkpoints:** try smaller baselines (`allenai/OLMo-1B`) or
  other sequence-classification models.
- **Adjust hyperparameters:** tweak `--max-length`, `--batch-size`, or
  `--learning-rate` to fit your GPU budget. Metrics printed at the end include
  RMSE/MAE for regression or accuracy for classification.
- **Keep it private:** add `--hub-private-repo` when pushing sensitive models
  back to the Hub.

The script is intentionally lightweight so you can extend it with custom metric
calculations, additional feature engineering (e.g., concatenating multiple text
fields), or logging backends such as Weights & Biases.