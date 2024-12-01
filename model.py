import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model='lstm', num_layers=1, dropout_p=0):
        super(MusicLSTM, self).__init__()
        self.model = model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(input_size, hidden_size)
        if self.model == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
        elif self.model == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            raise NotImplementedError

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.drop = nn.Dropout(p=dropout_p)

    def init_hidden(self, batch_size=1):
        """Initialize hidden states."""
        if self.model == 'lstm':
            self.hidden = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size)
            )
        elif self.model == 'gru':
            self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        return self.hidden

    def forward(self, x):
        """Forward pass."""
        # Ensure x is 2D (sequence length, batch size)
        if x.dim() > 2:
            x = x.squeeze()

        batch_size = 1 if x.dim() == 1 else x.size(0)
        x = x.long()

        # Embed the input
        embeds = self.embeddings(x)

        # Initialize hidden state if not already done
        if not hasattr(self, 'hidden'):
            self.init_hidden(batch_size)

        # Ensure embeds is 3D for RNN input (sequence length, batch size, embedding size)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(1)

        # RNN processing
        rnn_out, self.hidden = self.rnn(embeds, self.hidden)

        # Dropout and output layer
        rnn_out = self.drop(rnn_out.squeeze(1))
        output = self.out(rnn_out)

        return output
