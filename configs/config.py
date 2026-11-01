from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 2048
    n_embd : int = 384
    n_head : int = 6
    n_layer : int = 6
    vocab_size : int = 50304
    dropout_rate = 0.05
