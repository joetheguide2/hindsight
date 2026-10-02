# Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis

Research code, data and results for the paper:

> **Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis**
> (ZebraMap, Qwen-2.5-1.5B student ← DeepSeek-R1-Distill-LLaMA-8B teacher)

The 1.5B student is fine-tuned on teacher chain-of-thought traces generated with
the ground-truth label visible. Two students are compared:

| Name      | Training traces                | Role                          |
|-----------|--------------------------------|-------------------------------|
| Student   | raw teacher traces             | unfiltered ablation           |
| StudentF  | regex-filtered traces          | primary model (GT slots removed) |
| Teacher   | DeepSeek-R1-Distill-LLaMA-8B   | upper-bound reference         |
| Base      | Qwen-2.5-1.5B-Instruct         | pre-distillation baseline     |

See `paper/13_Hindsight_Guided_Rationale_.pdf` for the full paper.

## Repository layout

```
paper/                     Produced paper (PDF)
assets/
  docs/                    analysis_report.docx (earlier 1,976-case analysis)
  figures/                 Figures (candidate paper figures + earlier analysis)
src/
  data/                    Dataset construction / merging scripts
  training/                SFT + GRPO training scripts
  evaluation/              Model inference / evaluation scripts
  analysis/                Statistics, taxonomy and qualitative analyses
  inference.py, main.py    Utilities
data/
  raw/                     Source inputs (ZebraMap-derived CSV pieces)
  generated/               Processed/generated datasets (parquet/csv)
models/                    Final LoRA adapters / training runs
  student/                 Student   (results_finetuned_raw)
  studentF/                StudentF  (results_finetuned_clean)
  no_hindsight/            No-hindsight baseline (results_noh)
results/                   Evaluation outputs used for the paper's numbers
  base_and_student_raw/    Base + Student on the final 2,000-case eval set
  studentF_filtered/       StudentF (regex-filtered) evaluation
  base_tagged/             Base in zero-shot tag-prompting mode
  no_hindsight_baseline/   Student trained without label-visible traces
  teacher/                 DeepSeek teacher evaluation
archive/                   Everything not used for the final numbers (nothing deleted)
  data/ models/ results/ code/ caches/ misc/
```

## Paper artifacts → files

The final evaluation set is **`data/generated/eval_set_final.parquet`**
(2,000 cases, 250 diseases, 8 cases/disease).

| Paper artifact | Source file(s) |
|----------------|----------------|
| Base (Mentioned / Accuracy) | `results/base_and_student_raw/base_results.csv`, `results/base_tagged/` |
| Student (raw traces) | `results/base_and_student_raw/ft_results.csv` |
| StudentF (filtered traces) | `results/studentF_filtered/` (`regex_filtered.csv`, `ft_results.csv`) |
| Teacher | `results/teacher/deepseek_analysis_results.csv` |
| Contamination / reasoning-style analyses (Tables 3, 5; Figs 1–2) | `src/analysis/qualitative_analysis.py` |
| Accuracy / frequency analyses (Tables 1, 2, 4) | `src/analysis/quantitative_analysis.py` |
| Per-class / error breakdowns | `src/analysis/class_based_eval.py`, `src/analysis/eval_metrics.py`, `src/analysis/analysis*.py` |
| Figures | `assets/figures/` |

## Running scripts

Run commands **from the repository root** so the relative paths resolve, e.g.:

```bash
# Build base/cleaned datasets from the raw pieces
python src/data/dataset_creation.py

# Fine-tune StudentF (reads data/generated/cleaned_with_regex.parquet)
python src/training/train_fixed.py

# Evaluate (writes under results/)
python src/evaluation/eval.py
python src/evaluation/ft_eval.py

# Statistics / analyses
python src/analysis/quantitative_analysis.py
python src/analysis/qualitative_analysis.py
```

## Notes

- No file from the original repository was deleted during restructuring;
  unused/stale material was moved into `archive/` (grouped by type).
- `archive/` preserves the pre-restructure structure and is not part of the
  paper pipeline.
- Some scripts are historical (kept in `archive/code/` as `*.py~` backups or in
  `src/` for completeness) and may still reference archived paths.
