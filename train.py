import os
import sys
import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from model import MusicLSTM as MusicRNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import seq_to_tensor, load_vocab, save_vocab


def logger(active=True):
    """Simple logging utility."""
    def log(*args, **kwargs):
        if active:
            print(*args, **kwargs)
    return log

# Configuration
class Config:
    SAVE_EVERY = 20
    SEQ_SIZE = 25
    RANDOM_SEED = 11
    VALIDATION_SIZE = 0.15
    LR = 1e-3
    N_EPOCHS = 100
    NUM_LAYERS = 1
    HIDDEN_SIZE = 150
    DROPOUT_P = 0
    MODEL_TYPE = 'lstm'
    INPUT_FILE = 'data/music.txt'
    RESUME = False
    BATCH_SIZE = 1

# Utility functions
def tic():
    """Start timer."""
    return time.time()

def toc(start_time, msg=None):
    """Calculate elapsed time."""
    s = time.time() - start_time
    m = int(s / 60)
    if msg:
        return f'{m}m {int(s - (m * 60))}s {msg}'
    return f'{m}m {int(s - (m * 60))}s'

class DataLoader:
    def __init__(self, input_file, config):
        self.config = config
        self.char_idx, self.char_list = self._load_chars(input_file)
        self.data = self._load_data(input_file)
        self.train_idxs, self.valid_idxs = self._split_data()
        log = logger(True)
        log(f"Total songs: {len(self.data)}")
        log(f"Training songs: {len(self.train_idxs)}")
        log(f"Validation songs: {len(self.valid_idxs)}")
    
    def _load_chars(self, input_file):
        """Load unique characters from the input file."""
        with open(input_file, 'r') as f:
            char_idx = ''.join(set(f.read()))
        return char_idx, list(char_idx)

    def _load_data(self, input_file):
        """Load song data from input file."""
        with open(input_file, "r") as f:
            data, buffer = [], ''
            for line in f:
                if line == '<start>\n':
                    buffer += line
                elif line == '<end>\n':
                    buffer += line
                    data.append(buffer)
                    buffer = ''
                else:
                    buffer += line

        # Filter songs shorter than sequence size
        data = [song for song in data if len(song) > self.config.SEQ_SIZE + 10]
        return data

    def _split_data(self):
        """Split data into training and validation sets."""
        num_train = len(self.data)
        indices = list(range(num_train))

        np.random.seed(self.config.RANDOM_SEED)
        np.random.shuffle(indices)

        split_idx = int(np.floor(self.config.VALIDATION_SIZE * num_train))
        train_idxs = indices[split_idx:]
        valid_idxs = indices[:split_idx]

        return train_idxs, valid_idxs

    def rand_slice(self, data, slice_len=None):
        """Get a random slice of data."""
        if slice_len is None:
            slice_len = self.config.SEQ_SIZE

        d_len = len(data)
        s_idx = random.randint(0, d_len - slice_len)
        e_idx = s_idx + slice_len + 1
        return data[s_idx:e_idx]

    def seq_to_tensor(self, seq):
        """Convert sequence to tensor."""
        out = torch.zeros(len(seq)).long()
        for i, c in enumerate(seq):
            out[i] = self.char_idx.index(c)
        return out

    def song_to_seq_target(self, song):
        """Convert a song to sequence and target."""
        try:
            a_slice = self.rand_slice(song)
            seq = self.seq_to_tensor(a_slice[:-1])
            target = self.seq_to_tensor(a_slice[1:])
            return seq, target
        except Exception as e:
            print(f"Error in song_to_seq_target: {e}")
            print(f"Song length: {len(song)}")
            raise    

def train_model(config, data_loader, model, optimizer, loss_function):
    """Training loop for the model."""
    log = logger(True)
    time_since = tic()
    losses, v_losses = [], []

    for epoch in range(config.N_EPOCHS):
        # Training phase
        epoch_loss = 0
        model.train()

        for i, song_idx in enumerate(data_loader.train_idxs):
            try:
                seq, target = data_loader.song_to_seq_target(data_loader.data[song_idx])

                # Reset hidden state and gradients
                model.init_hidden()
                optimizer.zero_grad()

                # Forward pass
                outputs = model(seq)
                loss = loss_function(outputs, target)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                msg = f'\rTraining Epoch: {epoch}, {(i+1)/len(data_loader.train_idxs)*100:.2f}% iter: {i} Time: {toc(time_since)} Loss: {loss.item():.4f}'
                sys.stdout.write(msg)
                sys.stdout.flush()

            except Exception as e:
                log(f"Error processing song {song_idx}: {e}")
                continue

        print()
        losses.append(epoch_loss / len(data_loader.train_idxs))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, song_idx in enumerate(data_loader.valid_idxs):
                try:
                    seq, target = data_loader.song_to_seq_target(data_loader.data[song_idx])

                    # Reset hidden state
                    model.init_hidden()

                    # Forward pass
                    outputs = model(seq)
                    loss = loss_function(outputs, target)

                    val_loss += loss.item()

                    msg = f'\rValidation Epoch: {epoch}, {(i+1)/len(data_loader.valid_idxs)*100:.2f}% iter: {i} Time: {toc(time_since)} Loss: {loss.item():.4f}'
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                except Exception as e:
                    log(f"Error processing validation song {song_idx}: {e}")
                    continue

        print()
        v_losses.append(val_loss / len(data_loader.valid_idxs))

        # Checkpoint saving
        if epoch % config.SAVE_EVERY == 0 or epoch == config.N_EPOCHS - 1:
            log('=======> Saving..')
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': losses[-1],
                'v_loss': v_losses[-1],
                'losses': losses,
                'v_losses': v_losses,
                'epoch': epoch,
            }
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(model, f'checkpoint/ckpt_mdl_{config.MODEL_TYPE}_ep_{config.N_EPOCHS}_hsize_{config.HIDDEN_SIZE}_dout_{config.DROPOUT_P}.t{epoch}')

    return losses, v_losses

def plot_losses(losses, v_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.plot(v_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.show()

def generate_song(model, data_loader, prime_str='<start>', max_len=1000, temp=0.8):
    """Generate a new song using the trained model."""
    model.eval()
    model.init_hidden()
    creation = prime_str
    char_idx, char_list = load_vocab()

    # Build up hidden state
    prime = seq_to_tensor(creation, char_idx)
    print(prime)

    with torch.no_grad():
        for _ in range(len(prime)-1):
            _ = model(prime[_:_+1])

        # Generate rest of sequence
        for _ in range(max_len):
            last_char = prime[-1:]
            out = model(last_char).squeeze()

            out = torch.exp(out/temp)
            dist = out / torch.sum(out)

            # Sample from distribution
            next_char_idx = torch.multinomial(dist, 1).item()
            next_char = char_idx[next_char_idx]

            creation += next_char
            prime = torch.cat([prime, torch.tensor([next_char_idx])], dim=0)

            if creation[-5:] == '<end>':
                break

    return creation

def main():
    """Main execution function."""
    # Set up configuration and data
    global model, data_loader
    config = Config()
    data_loader = DataLoader(config.INPUT_FILE, config)

    # Model setup
    in_size = out_size = len(data_loader.char_idx)
    model = MusicRNN(
        in_size,
        config.HIDDEN_SIZE,
        out_size,
        config.MODEL_TYPE,
        config.NUM_LAYERS,
        config.DROPOUT_P
    )

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_function = nn.CrossEntropyLoss()

    # Train the model
    losses, v_losses = train_model(config, data_loader, model, optimizer, loss_function)

    # Plot losses
    plot_losses(losses, v_losses)
    save_vocab(data_loader)

    # Generate a song
    generated_song = generate_song(model, data_loader)
    print("Generated Song:", generated_song)

if __name__ == "__main__":
    main()