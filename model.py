import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MusicLSTM(nn.Module):
    def __int__(self, input_size, hidden_size, output_size, model='lstm', num_layers=1,DROPOUT_P=0):
        super(MusicLSTM, self).__init__()
        self.DROPOUT_P = DROPOUT_P
        self.model = model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        if self.model == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
        elif self.model == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            raise NotImplementedError
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.drop = nn.Dropout(p=self.DROPOUT_P)
    
    def init_hidden(self):
        if self.model =='lstm':
            self.hidden = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                           Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        elif self.model == 'gru':
            self.hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
    
    def forward(self, seq):
        embeds = self.embedding(seq.view(1, -1))
        rnn_out, self.hidden = self.rnn(embeds.view(1,1,-1), self.hidden)
        rnn_out = self.drop(rnn_out)
        output = self.out(rnn_out.view(1, -1))
        return output
