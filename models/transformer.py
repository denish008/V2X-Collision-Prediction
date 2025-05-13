import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class TransformerConfig:
    input_dim: int  # Number of features
    hidden_dim: int = field(default=64) # Hidden dimension
    num_layers: int = field(default=2)  # Number of encoder layers
    num_heads: int = field(default=2)   # Number of attention heads
    dropout_prob: float = field(default=0.1)
    output_dim: int = field(default=1) 
    feedforward_dim: int = field(default=128) #Feedforward dimension. A best value that is greater than hidden value.


class Transformer(nn.Module):
    def __init__(self, net_config: TransformerConfig):
        super().__init__()
        self.net_config = net_config
        self.input_dim = net_config.input_dim

        self.input_projection = nn.Linear(net_config.input_dim, net_config.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=net_config.hidden_dim,
            nhead=net_config.num_heads,
            dropout=net_config.dropout_prob,
            dim_feedforward=net_config.feedforward_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=net_config.num_layers
        )
        self.linear = nn.Linear(net_config.hidden_dim, net_config.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  
        x = self.linear(x) 
        x = self.sigmoid(x)
        return x