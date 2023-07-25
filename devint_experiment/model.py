import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from devint_experiment.constants import CONTEXT_WINDOW_LENGTH, DEVICE


class PositionalEncoding(nn.Module):

    def __init__(self, dim_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10_000.0) / dim_model))
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.sin(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape src: ``[seq_len, batch_size]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, num_tokens: int, dim_model: int, num_heads: int,
                 dim_hidden: int, num_layers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.pos_encoder.requires_grad = True
        encoder_layers = TransformerEncoderLayer(dim_model, num_heads, dim_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.transformer_encoder.requires_grad = True
        self.encoder = nn.Embedding(num_tokens, dim_model)
        self.encoder.requires_grad = True
        self.dim_model = dim_model
        self.decoder = nn.Linear(dim_model, num_tokens)
        self.decoder.requires_grad = True
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, batch_size]``
            
        Returns:
            output Tensor of shape ``[seq_len, batch_size, num_tokens]``
        """

        src = self.encoder(src) * math.sqrt(self.dim_model)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)

        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
