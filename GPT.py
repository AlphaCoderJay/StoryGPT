import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.dropout = nn.Dropout(config.dropout_rate)


    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)

        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)

        y  = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        y = self.c_proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.layer1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.layer2 = nn.Linear(config.n_embd * 4, config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)


    def forward(self,x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        x = self.dropout(x)

        return x




class Block(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)


    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x



class GPT2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self,idx,targets=None):
        B,T = idx.size()

        pos = torch.arange(0,T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        x = self.dropout(x)
        for block in self.transformer.h:
            x = block(x)


        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):

            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


