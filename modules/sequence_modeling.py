import torch.nn as nn
import torch

from .transformers import SinPositionalEncoding, PositionalEmbedding

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class TorchEncoder(nn.Module):
    def __init__(self, 
            d_model: int, num_layers: int,
            seq_length: int, learnable_embeddings: bool,
            dropout: float
        ) -> None:
        super().__init__()
        self.model = nn.TransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4,
                batch_first=True,dropout=dropout
            ), 
            num_layers=num_layers
        )
        if learnable_embeddings:
            self.position_embeddings = PositionalEmbedding(learnable=True, num_embeddings=seq_length, embedding_dim=d_model,dropout=dropout)
        else:
            self.position_embeddings = SinPositionalEncoding(d_model=d_model, max_len=seq_length,dropout=dropout)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        embedded_input = self.position_embeddings(input)
        encoder_out = self.model(embedded_input)
        return encoder_out