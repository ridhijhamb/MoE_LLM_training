from .moe import MoE  

class MoETransformerLayer(nn.Module):  
    def __init__(self, dim, num_heads, num_experts=4, top_k=2):  
        super().__init__()  
        self.attention = nn.MultiheadAttention(dim, num_heads)  
        self.moe = MoE(dim, num_experts, top_k)  
        self.norm1 = nn.LayerNorm(dim)  
        self.norm2 = nn.LayerNorm(dim)  

    def forward(self, x):  
        # Self-attention  
        attn_out, _ = self.attention(x, x, x)  
        x = self.norm1(x + attn_out)  

        # MoE FFN  
        moe_out = self.moe(x)  
        x = self.norm2(x + moe_out)  

        return x  