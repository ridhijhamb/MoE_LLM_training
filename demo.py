import torch
from transformers import GPT2Tokenizer
from models.transformer import MoETransformer

# Load models
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
moe_model = torch.load("moe_model.pth")["model"].eval().cuda()

# Interactive generation
while True:
    text = input("Enter prompt: ")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = moe_model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))