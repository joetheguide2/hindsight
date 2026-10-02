# Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis

Code, data and results for the paper:

> **Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis**
> ZebraMap · Qwen-2.5-1.5B student ← DeepSeek-R1-Distill-LLaMA-8B teacher

The 1.5B student is fine-tuned on teacher chain-of-thought traces that were
generated with the ground-truth label visible. Two students are compared:

| Name     | Training traces               | Role                             |
|----------|-------------------------------|----------------------------------|
| Student  | raw teacher traces            | unfiltered ablation              |
| StudentF | regex-filtered traces         | primary model (GT slots removed) |
| Teacher  | DeepSeek-R1-Distill-LLaMA-8B  | upper-bound reference            |
| Base     | Qwen-2.5-1.5B-Instruct        | pre-distillation baseline        |

The full paper is in `paper/13_Hindsight_Guided_Rationale_.pdf`.

## Tracked contents

This repository intentionally tracks only the final, non-redoable artifacts.
Everything else is either regenerable from these files or is a historical / failed
run kept only on disk and excluded via `.gitignore` (see below).

```
paper/13_Hindsight_Guided_Rationale_.pdf        Produced paper
assets/docs/analysis_report.docx                Earlier (1,976-case) analysis report
data/generated/eval_set_final.parquet           Final evaluation set (2,000 cases, 250 diseases)
results/base_and_student_raw/ft_results.csv     Student predictions on the eval set
results/studentF_filtered/ft_results.csv        StudentF predictions on the eval set
src/data/dataset_creation.py                    Builds base/cleaned from raw CSV pieces
src/training/train_fixed.py                     Final SFT training script (the only training entrypoint)
src/evaluation/eval.py                          Base + no-hindsight student evaluation
src/evaluation/ft_eval.py                       Student / StudentF evaluation
src/analysis/quantitative_analysis.py           Accuracy / frequency statistics (Tables 1, 2, 4)
src/analysis/qualitative_analysis.py            Contamination + reasoning-style analyses (Tables 3, 5)
src/analysis/class_based_eval.py                Per-class breakdowns
src/analysis/eval_metrics.py                    Accuracy / agreement metrics
pyproject.toml, uv.lock, .gitignore, README.md
```

## Paper artifacts → files

The final evaluation set is `data/generated/eval_set_final.parquet`
(2,000 cases, 250 diseases, 8 cases/disease).

| Paper artifact | Source file(s) |
|----------------|----------------|
| Base (Mentioned / Accuracy) | `results/base_and_student_raw/base_results.csv` *(gitignored, regenerable)* |
| Student (raw traces) | `results/base_and_student_raw/ft_results.csv` |
| StudentF (filtered traces) | `results/studentF_filtered/ft_results.csv` |
| Teacher | `results/teacher/deepseek_analysis_results.csv` *(gitignored, regenerable)* |
| Contamination / reasoning-style (Tables 3, 5; Figs 1–2) | `src/analysis/qualitative_analysis.py` |
| Accuracy / frequency (Tables 1, 2, 4) | `src/analysis/quantitative_analysis.py` |
| Per-class / agreement metrics | `src/analysis/class_based_eval.py`, `src/analysis/eval_metrics.py` |

## Intentionally omitted (gitignored, redoable)

These are present on disk but **not** committed, because they are large
(model weights), redundant, or historical/failed runs. The paper's figures are
not committed either — the paper PDF already contains them.

- `models/`, `archive/`, `runs/`, `*.safetensors`, `*.pt`, `*.bin`, `*.gguf` — model weights and training runs.
- `data/raw/`, and every generated parquet/csv except the eval set listed above —
  raw CSV pieces and intermediate datasets. This includes the final training
  corpus `data/generated/cleaned_with_regex.parquet` (27,501 samples), which is
  regenerable and exceeds GitHub's 100 MB file limit.
- All `results/` except the two Student/StudentF `ft_results.csv` files —
  including teacher, base, tag-prompted and no-hindsight runs.
- `assets/figures/` — all analysis figures (superseded by the paper).
- One-off / superseded / data-inspection scripts:
  `src/main.py`, `src/inference.py`, `src/training/train.py`,
  `src/training/grposcript.py` (GRPO was never used),
  `src/training/save_trained_model.py`, `src/evaluation/eval_batched.py`,
  `src/evaluation/evaluate_new.py`, `src/evaluation/eval_deepseek.py`,
  `src/analysis/pandasyaar.py`, `src/analysis/analysis.py`,
  `src/analysis/analysis2.py`, `src/analysis/analysis3.py`,
  `src/analysis/plot.py`, `src/data/checklen.py`, `src/data/concat_all.py`,
  `src/data/jsonput.py`, `src/data/merge.py`.

The repo therefore contains a number of exploratory/failed scripts and stale
datasets on disk; only the files listed under **Tracked contents** are meant to
be relied upon. The omitted scripts that simply concatenate files, inspect a
dataset, or duplicate earlier work are marked above.

## Running

Run commands **from the repository root** so the relative paths resolve:

```bash
# 1. Build the base/cleaned datasets from the raw CSV pieces (in data/raw)
python src/data/dataset_creation.py

# 2. Fine-tune StudentF  (reads data/generated/cleaned_with_regex.parquet;
#    writes the LoRA adapter + checkpoints under models/studentF)
python src/training/train_fixed.py

# 3. Evaluate (writes under results/)
python src/evaluation/eval.py
python src/evaluation/ft_eval.py

# 4. Statistics / analyses
python src/analysis/quantitative_analysis.py
python src/analysis/qualitative_analysis.py
```

## Notes

- No original file was deleted while restructuring; unused material was moved
  into `archive/` (grouped by type) and is gitignored.
- Data paths in the tracked scripts are written relative to the repo root.
