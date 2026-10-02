import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from unsloth import FastLanguageModel

# 1. Load your tokenizer (must match your model)
model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit" # Change if needed
max_seq_length = 8192 # Use a high number just for the audit
tokenizer = FastLanguageModel.get_tokenizer(model_name)

# 2. Load your dataset
# Replace with your actual loading logic from train_fixed.py
ds = load_dataset("json", data_files="your_data.jsonl", split="train")

# 3. Calculate token lengths
def get_token_length(example):
    # If using ChatML/standard text field
    return {"token_count": len(tokenizer.encode(example["text"]))}

print("Calculating token counts...")
lengths_ds = ds.map(get_token_length, batched=False)
counts = np.array(lengths_ds["token_count"])

# 4. Show Statistics
print("\n--- TOKEN DISTRIBUTION STATS ---")
print(f"Total samples: {len(counts)}")
print(f"Min length:    {counts.min()}")
print(f"Mean length:   {counts.mean():.1f}")
print(f"Median:        {np.median(counts)}")
print(f"90th %-tile:   {np.percentile(counts, 90)}")
print(f"95th %-tile:   {np.percentile(counts, 95)}")
print(f"99th %-tile:   {np.percentile(counts, 99)}")
print(f"Max length:    {counts.max()}")

# 5. Visualize
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=50, color='skyblue', edgecolor='black')
plt.title("Dataset Token Length Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Samples")
plt.axvline(2048, color='red', linestyle='--', label='2048 Limit')
plt.axvline(4096, color='orange', linestyle='--', label='4096 Limit')
plt.legend()
plt.show()
