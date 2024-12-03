import os
import sys
import time
import json
import torch

# Only do the function below if verbose
def logger(verbose):
    def log(*msg):
        if verbose: print(*msg)
    return log


last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    sys.stdout.write('\r')
    #if current < total-1:
    #
    #else:
        #sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_vocab(data_loader, vocab_filename="config/vocab.json"):
    """Save vocabulary to a JSON file."""
    vocab = {
        'char_idx': data_loader.char_idx,
        'char_list': data_loader.char_list
    }
    with open(vocab_filename, 'w') as f:
        json.dump(vocab, f)

def load_vocab(vocab_filename='config/vocab.json'):
    with open(vocab_filename, 'r') as f:
        vocab = json.load(f)
    return vocab['char_idx'], vocab['char_list']

def seq_to_tensor(seq, char_idx):
        """Convert sequence to tensor."""
        out = torch.zeros(len(seq)).long()
        for i, c in enumerate(seq):
            out[i] = char_idx.index(c)
        return out