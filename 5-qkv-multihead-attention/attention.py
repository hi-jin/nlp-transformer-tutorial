import math
from typing import TypedDict, Optional
import torch
from torch import nn


class AttentionConfig(TypedDict):
    dim_embed: int
    num_heads: int


class Attention(nn.Module):
    def __init__(
        self,
        config: AttentionConfig,
    ):
        super().__init__()
        self.config = config
        
        assert self.config['dim_embed'] % self.config['num_heads'] == 0

        self.dim_k = self.config['dim_embed'] // self.config['num_heads']
        
        self.query = nn.Linear(self.config['dim_embed'], self.config['dim_embed'])
        self.key = nn.Linear(self.config['dim_embed'], self.config['dim_embed'])
        self.value = nn.Linear(self.config['dim_embed'], self.config['dim_embed'])
        self.to_output = nn.Linear(self.config['dim_embed'], self.config['dim_embed'])
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        q = self.__split_heads(self.query(query))
        k = self.__split_heads(self.key(key))
        v = self.__split_heads(self.value(value))

        reactivity = torch.matmul(q, k.transpose(-1, -2)) # [batch_size, num_heads, query_len, key_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = 1 - attention_mask
            attention_mask = attention_mask * -float("inf")
            reactivity += attention_mask
        attention_score = (reactivity / math.sqrt(self.dim_k)).softmax(dim=-1)

        blended_vector = torch.matmul(attention_score, v) # [batch_size, num_heads, query_len, dim_k]
        blended_vector = self.__join_heads(blended_vector) # [batch_size, query_len, dim_embed]
        blended_vector = self.to_output(blended_vector)

        return blended_vector, attention_score

    def __split_heads(
        self,
        x: torch.Tensor,
    ):
        batch_size, seq_len, dim_embed = x.shape
        x = x.view(batch_size, seq_len, self.config['num_heads'], self.dim_k).contiguous()
        x = x.transpose(1, 2)
        return x

    def __join_heads(
        self,
        x: torch.Tensor,
    ):
        batch_size, num_heads, seq_len, dim_k = x.shape
        x = x.transpose(1, 2)
        x = x.view(batch_size, seq_len, self.config['dim_embed']).contiguous()
        return x
