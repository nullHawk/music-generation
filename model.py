import torch
import torch.nn as nn

class MusicGenerationLSTM(nn.Module):
    def __init__(self, input_size,  hidden_size, n_layers):
        super(MusicGenerationLSTM, self).__init__()