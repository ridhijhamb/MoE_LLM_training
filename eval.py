import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.transformer import MoETransformer  # Your custom MoE model

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CONFIG = {
    "dim": 768,
    "num_heads": 12,
    "num_layers": 6,
    "num_experts": 4,  # MoE-only
    "top_k": 2         # MoE-only
}
TEST_TEXT = "The quick brown fox jumps over the lazy dog"
NUM_TOKENS_TO_GENERATE = 100

# --- Initialize Models ---
def load_models():
    """Load both MoE and dense models for comparison"""
    # Your MoE Model
    moe_model = MoETransformer(
        dim=MODEL_CONFIG["dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_experts=MODEL_CONFIG["num_experts"],
        top_k=MODEL_CONFIG["top_k"]
    ).to(DEVICE)
    
    # Comparable Dense Model (similar total capacity)
    dense_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)  # 124M params
    
    # Alternative: Your custom dense Transformer if available
    # from models.transformer import DenseTransformer
    # dense_model = DenseTransformer(dim=768, num_heads=12, num_layers=6).to(DEVICE)
    
    return moe_model, dense_model

# --- Benchmarking Functions ---
def benchmark_speed(model, tokenizer):
    """Measure tokens generated per second"""
    inputs = tokenizer(TEST_TEXT, return_tensors="pt").to(DEVICE)
    
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10)
    
    # Timed generation
    start = time.time()
    _ = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE)
    elapsed = time.time() - start
    
    return NUM_TOKENS_TO_GENERATE / elapsed

def benchmark_memory(model, tokenizer):
    """Measure peak GPU memory usage during inference"""
    inputs = tokenizer(TEST_TEXT, return_tensors="pt").to(DEVICE)
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

def calculate_perplexity(model, tokenizer, dataset):
    """Compute perplexity on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in dataset:
        inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return torch.exp(torch.tensor(avg_loss)).item()

def count_parameters(model):
    """Return total and trainable parameters (in millions)"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable

# --- Main Evaluation ---
def main():
    # Load models and tokenizer
    moe_model, dense_model = load_models()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load validation dataset (small subset for demo)
    from datasets import load_dataset
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10%]")
    
    # Run benchmarks
    results = {}
    
    for name, model in [("MoE", moe_model), ("Dense", dense_model)]:
        model.eval()
        with torch.no_grad():
            results[name] = {
                "speed (tok/s)": benchmark_speed(model, tokenizer),
                "memory (MB)": benchmark_memory(model, tokenizer),
                "perplexity": calculate_perplexity(model, tokenizer, test_data),
                "params (M)": count_parameters(model)[0]
            }
    
    # Print results
    print("\n=== Benchmark Results ===")
    print(f"Test config: top_k={MODEL_CONFIG['top_k']}, experts={MODEL_CONFIG['num_experts']}")
    print(f"Device: {DEVICE}\n")
    
    for metric in ["speed (tok/s)", "memory (MB)", "perplexity", "params (M)"]:
        print(f"{metric}:")
        for model_name in results:
            print(f"  {model_name}: {results[model_name][metric]:.1f}")

if __name__ == "__main__":
    main()