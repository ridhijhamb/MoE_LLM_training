# models/moe.py  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class MoE(nn.Module):  
    def __init__(self, dim, num_experts=4, top_k=2):  
        super().__init__()  
        self.dim = dim  
        self.num_experts = num_experts  
        self.top_k = top_k  

        # Experts are simple FFNs  
        self.experts = nn.ModuleList([  
            nn.Sequential(  
                nn.Linear(dim, dim * 4),  
                nn.ReLU(),  
                nn.Linear(dim * 4, dim)  
            ) for _ in range(num_experts)  
        ])  

        # Router decides which experts to use  
        self.router = nn.Linear(dim, num_experts)  

    def forward(self, x):  
        # x shape: [batch_size, seq_len, dim]  
        batch_size, seq_len, dim = x.shape  

        # Get router logits  
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]  

        # Top-k expert selection  
        top_k_weights, top_k_experts = torch.topk(  
            router_logits, self.top_k, dim=-1  
        )  
        top_k_weights = F.softmax(top_k_weights, dim=-1)  

        # Initialize output  
        out = torch.zeros_like(x)  

        # For each expert, process the tokens assigned to it  
        for expert_id in range(self.num_experts):  
            expert_mask = (top_k_experts == expert_id)  
            if expert_mask.any():  
                expert_input = x[expert_mask]  
                expert_output = self.experts[expert_id](expert_input)  
                out[expert_mask] += expert_output * top_k_weights[expert_mask]  

        return out  
    
    
    def load_balancing_loss(router_logits, top_k_experts):  
        # router_logits: [batch_size, seq_len, num_experts]  
        # top_k_experts: [batch_size, seq_len, top_k]  

        # Compute fraction of tokens per expert  
        expert_counts = torch.bincount(  
            top_k_experts.flatten(),  
            minlength=router_logits.size(-1)  
        )  
        expert_fraction = expert_counts / expert_counts.sum()  

        # Router's probability mass per expert  
        router_probs = torch.softmax(router_logits, dim=-1)  
        router_fraction = router_probs.mean(dim=[0, 1])  

        # Load balancing loss  
        return (expert_fraction * router_fraction).sum()  