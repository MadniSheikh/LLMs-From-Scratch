import torch 
import torch.nn as nn

class CasualAttention(nn.Module):
    '''' 
    Cross attention that masks the future token to attend while training
    '''
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, X):
        b, num_tokens, d_in = X.shape
        keys = self.W_key(X)
        queries = self.W_query(X)
        values = self.W_value(X)

        attn_score = queries @ keys.transpose(1, 2)
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_score/keys.shape[-1]**0.5, dim = -1) 
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

# MultiHeadAttention wrapper on CasualAttention
class MultiHeadAttention(nn.Module):
    ''' 
    Attention divided into heads so that LLM can produce more perspective
    '''
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.out_proj = nn.Linear(d_out, d_out)

        assert d_out % num_heads == 0

        head_dim = d_out // num_heads

        self.heads = nn.ModuleList(
            [CasualAttention(d_in, head_dim, context_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, X):
        context = torch.cat(
            [head(X) for head in self.heads],
            dim=-1
        )
        return self.out_proj(context)