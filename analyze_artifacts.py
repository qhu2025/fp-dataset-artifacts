import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import os

def analyze_data():
    model_path = "./output_hypothesis_only" 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}.")
        return

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load the same 50k data used for the hard subset
    print("Loading SNLI Training Data (First 50k)...")
    dataset = load_dataset("snli", split="train[:50000]")
    
    easy_examples = [] 
    
    # Classify examples
    print("Classifying examples for forensics...")
    for example in tqdm(dataset):
        if example['label'] == -1: continue
        
        # Tokenize ONLY hypothesis
        inputs = tokenizer(
            example['hypothesis'], 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        pred = logits.argmax().item()
        
        # 0: Entailment, 1: Neutral, 2: Contradiction
        if pred == example['label']:
            easy_examples.append({
                'hypothesis': example['hypothesis'].lower(),
                'label': example['label'] 
            })

    # Calculate negation bias
    easy_contradictions = [x for x in easy_examples if x['label'] == 2]
    
    negation_words = ['no', 'not', 'never', 'nobody', 'nothing', "n't"]
    
    # Count # easy contradictions containing at least 1 negation word
    neg_count = 0
    for ex in easy_contradictions:
        if any(w in ex['hypothesis'].split() for w in negation_words):
            neg_count += 1
            
    print("\n" + "="*40)
    print("FORENSICS ANALYSIS RESULTS")
    print("="*40)
    print(f"Total 'Easy' examples found: {len(easy_examples)}")
    print(f"Total 'Easy' Contradictions: {len(easy_contradictions)}")
    
    if len(easy_contradictions) > 0:
        percentage = neg_count / len(easy_contradictions)
        print(f"Percentage containing negation words: {percentage:.1%}")
        print(f"(Raw count: {neg_count} out of {len(easy_contradictions)})")
    else:
        print("No easy contradictions found.")
    print("="*40)

if __name__ == "__main__":
    analyze_data()