import torch
from torch import nn
from dataclasses import dataclass, field


@dataclass
class LSTMConfig:
    input_dim: int 
    hidden_dim: int 
    num_layers: int 
    output_dim: int
    dropout_prob: float = field(default=0.1)


class LSTM(nn.Module):
    def __init__(self, net_config: LSTMConfig):
        super(LSTM, self).__init__()
        self.net_config = net_config
        self.input_dim = self.net_config.input_dim
        self.hidden_dim = self.net_config.hidden_dim
        self.num_layers = self.net_config.num_layers
        self.output_dim = self.net_config.output_dim
        self.dropout_prob = self.net_config.dropout_prob

        self.lstm = nn.LSTM(
            self.input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True, 
            dropout=self.dropout_prob
        ) 
        self.dropout = nn.Dropout(self.dropout_prob)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid() # For binary classification

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=x.device
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=x.device
        ).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # out.shape = (batch_size, seq_length, hidden_size)
        out = self.dropout(out[:, -1, :])  # Only take the last time step's output
        out = self.linear(out)
        out = self.sigmoid(out)  # Apply sigmoid for binary classification
        return out