import os
import gc
import re
import random
import ast
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import warnings
from transformers import logging as hf_logging

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
hf_logging.set_verbosity_error()

# -------------------- Configuration --------------------
CSV_PATH = "data/generated/eval_truncated_subset_raw.csv"
SAMPLE_SIZE = None
RANDOM_SEED = 42
SYSTEM_PROMPT = "You are a medical diagnostic expert."
MAX_SEQ_LENGTH = 8192
LORA_RANK = 64
BATCH_SIZE = 1

GEN_KWARGS = {
    "max_new_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "use_cache": True,
}

OUTPUT_DIR = "results/ft_only_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FT_CSV = os.path.join(OUTPUT_DIR, "ft_results.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
PLOT_ACCURACY = os.path.join(OUTPUT_DIR, "accuracy.png")
PLOT_TAGS = os.path.join(OUTPUT_DIR, "tag_presence.png")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# -------------------- Load and sample data --------------------
df = pd.read_csv(CSV_PATH)
assert "CaseSummary" in df.columns and "Disease" in df.columns, (
    "Parquet must contain 'CaseSummary' and 'Disease' columns"
)

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
print(f"Using {len(df)} cases for evaluation.")


# -------------------- Helper functions --------------------
def parse_synonyms(syn_str):
    if pd.isna(syn_str) or not isinstance(syn_str, str):
        return []
    return [s.strip() for s in syn_str.split(",") if s.strip()]


def disease_in_response(response, disease, synonyms):
    response_lower = response.lower()
    if disease and disease.lower() in response_lower:
        return True
    for syn in synonyms:
        if syn.lower() in response_lower:
            return True
    return False


def load_results_from_csv(csv_path):
    df_csv = pd.read_csv(csv_path)
    results = []
    for _, row in df_csv.iterrows():
        try:
            synonyms = (
                ast.literal_eval(row["synonyms"]) if pd.notna(row["synonyms"]) else []
            )
        except (SyntaxError, ValueError):
            synonyms = []
        results.append(
            {
                "case_idx": row["case_idx"],
                "true_disease": row["true_disease"],
                "synonyms": synonyms,
                "response": row["response"],
                "correct": bool(row["correct"]),
                "has_think": bool(row["has_think"]),
                "has_diagnose": bool(row["has_diagnose"]),
            }
        )
    return results


def evaluate_model(
    model,
    tokenizer,
    df,
    model_name,
    batch_size=BATCH_SIZE,
    output_csv=None,
    overwrite=True,
):
    results = []

    if output_csv:
        if overwrite and os.path.exists(output_csv):
            os.remove(output_csv)
            print(f"  (Existing {output_csv} removed, starting fresh)")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(df), batch_size), desc=f"Evaluating {model_name}"):
        batch_df = df.iloc[i : i + batch_size]
        prompts = []
        indices = []

        for idx, row in batch_df.iterrows():
            case = row["CaseSummary"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Look at the patient case and diagnose them. Case Summary: {case}",
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)
            indices.append(idx)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **GEN_KWARGS)

        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        batch_results = []
        for j, idx in enumerate(indices):
            row = df.loc[idx]
            true_disease = row["Disease"]
            synonyms = parse_synonyms(row.get("Synonyms", ""))
            response = responses[j].strip()
            correct = disease_in_response(response, true_disease, synonyms)
            has_think = "</think>" in response
            has_diagnose = "<diagnose>" in response or "<diagnosis>" in response
            res = {
                "case_idx": idx,
                "true_disease": true_disease,
                "synonyms": synonyms,
                "response": response,
                "correct": correct,
                "has_think": has_think,
                "has_diagnose": has_diagnose,
            }
            batch_results.append(res)
            results.append(res)

        if output_csv:
            batch_out = pd.DataFrame(batch_results)
            header = not os.path.exists(output_csv)
            batch_out.to_csv(output_csv, mode="a", header=header, index=False)

    return results


# -------------------- Fine-tuned Model Evaluation --------------------
ft_results = None
ft_model_loaded = False

if os.path.exists(FT_CSV):
    existing_df = pd.read_csv(FT_CSV)
    if len(existing_df) == len(df):
        print("\n" + "=" * 50)
        print("Fine-tuned model results already complete. Loading from CSV.")
        ft_results = load_results_from_csv(FT_CSV)
    else:
        print("\n" + "=" * 50)
        print(
            f"Fine-tuned model results partial ({len(existing_df)}/{len(df)}). Evaluating remaining cases."
        )
        processed_indices = set(existing_df["case_idx"])
        remaining_df = df[~df.index.isin(processed_indices)]

        print("Loading fine-tuned model...")
        ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
            model_name="models/student/finetuned_raw_lora",
            load_in_4bit=True,
            fast_inference=False,
            max_lora_rank=LORA_RANK,
            local_files_only=False,
            max_seq_length=MAX_SEQ_LENGTH,  # <--- ADD THIS LINE
            device_map="auto",
        )
        ft_model = FastLanguageModel.for_inference(ft_model)
        ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="qwen2.5")
        ft_model_loaded = True

        _ = evaluate_model(
            ft_model,
            ft_tokenizer,
            remaining_df,
            "Fine-tuned Model (resume)",
            output_csv=FT_CSV,
            overwrite=False,
        )
        ft_results = load_results_from_csv(FT_CSV)
else:
    print("\n" + "=" * 50)
    print("No fine-tuned results found. Evaluating full set.")
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/studentF/finetuned_clean_lora",
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=LORA_RANK,
        local_files_only=False,
        max_seq_length=MAX_SEQ_LENGTH,  # <--- ADD THIS LINE
        device_map="auto",
    )
    ft_model = FastLanguageModel.for_inference(ft_model)
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="qwen2.5")
    ft_model_loaded = True

    ft_results = evaluate_model(
        ft_model,
        ft_tokenizer,
        df,
        "Fine-tuned Model",
        output_csv=FT_CSV,
        overwrite=True,
    )

if ft_model_loaded:
    del ft_model, ft_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# -------------------- Analyze and save results --------------------
ft_df = pd.DataFrame(ft_results)
ft_df.to_csv(OUTPUT_CSV, index=False)

ft_acc = ft_df["correct"].mean()
ft_think_pct = ft_df["has_think"].mean() * 100
ft_diagnose_pct = ft_df["has_diagnose"].mean() * 100

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Fine-tuned model accuracy: {ft_acc * 100:.2f}%")
print(f"Fine-tuned </think> tag:   {ft_think_pct:.2f}%")
print(f"Fine-tuned <diagnose> tag: {ft_diagnose_pct:.2f}%")
print(f"\nDetailed results saved to {OUTPUT_CSV}")

# -------------------- Plots --------------------
sns.set_style("whitegrid")

# Accuracy plot
plt.figure(figsize=(5, 5))
bar = plt.bar(["Fine-tuned Model"], [ft_acc], color=["#ff7f0e"])
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.text(
    bar[0].get_x() + bar[0].get_width() / 2,
    bar[0].get_height() + 0.02,
    f"{ft_acc * 100:.1f}%",
    ha="center",
    va="bottom",
)
plt.title("Diagnosis Accuracy")
plt.tight_layout()
plt.savefig(PLOT_ACCURACY, dpi=150)
print(f"Accuracy plot saved to {PLOT_ACCURACY}")

# Tag presence plot
plt.figure(figsize=(6, 5))
tags = ["</think>", "<diagnose>"]
tag_pcts = [ft_think_pct, ft_diagnose_pct]
bars = plt.bar(tags, tag_pcts, color=["#2ca02c", "#d62728"])
plt.ylabel("Presence (%)")
plt.ylim(0, 100)
for bar, pct in zip(bars, tag_pcts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
    )
plt.title("Tag Presence in Fine-tuned Model")
plt.tight_layout()
plt.savefig(PLOT_TAGS, dpi=150)
print(f"Tag presence plot saved to {PLOT_TAGS}")

print("\nEvaluation complete.")
