import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from models.transformer import MoETransformer  # Your custom MoE model

# --- Config ---
MODEL_CONFIG = {
    "dim": 768,
    "num_heads": 12,
    "num_layers": 6,
    "num_experts": 4,
    "top_k": 2
}
BATCH_SIZE = 4
GRADIENT_ACCUM_STEPS = 8  # Simulate larger batch size
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load dataset
dataset = load_dataset("openwebtext", split="train[:1%]")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model and optimizer
model = MoETransformer(**MODEL_CONFIG).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --- Training Loop ---
model.train()
for epoch in range(3):  # Example: 3 epochs
    for step, batch in enumerate(dataloader):
        # Tokenize batch
        inputs = tokenizer(
            batch["text"], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).input_ids.to(DEVICE)
        
        # Forward pass
        outputs = model(inputs)
        
        # Loss calculation
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # Add MoE load balancing loss (if your model returns it)
        if hasattr(outputs, "router_logits"):
            loss += 0.01 * load_balancing_loss(outputs.router_logits, outputs.top_k_experts)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.3f}")

# --- Save Model ---
torch.save({
    "model_state_dict": model.state_dict(),
    "config": MODEL_CONFIG
}, "moe_model.pth")