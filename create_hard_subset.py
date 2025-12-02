import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import os


def create_hard_dataset():
    # Load the hypothesis-only model
    model_path = "./output_hypothesis_only"
    
    if not os.path.exists(model_path):
        print("Model not found. Please run Part 1 training first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} to {device}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load first 50k training data 
    print("Loading SNLI Training Data (First 50k)...")
    dataset = load_dataset("snli", split="train[:50000]")

    print("Filtering out 'Easy' examples...")
    hard_examples = []
    total = 0
    kept = 0
    
    for example in tqdm(dataset):
        if example['label'] == -1: continue
        
        # Tokenize ONLY the hypothesis
        inputs = tokenizer(
            example['hypothesis'], 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predicted_class = logits.argmax().item()

        if predicted_class != example['label']:
            hard_examples.append({
                "premise": example['premise'],
                "hypothesis": example['hypothesis'],
                "label": example['label']
            })
            kept += 1
        total += 1

    # Save the new dataset
    print(f"Stats: Processed {total}. Kept {kept} ({kept/total:.1%} kept).")
    with open("hard_snli.json", "w") as f:
        for ex in hard_examples:
            json.dump(ex, f)
            f.write('\n')
    print("Created 'hard_snli.json'")

if __name__ == "__main__":
    create_hard_dataset()