import glob
import pickle
import torch
import numpy as np
from music21 import converter, instrument, note, chord

def get_notes(data_path):
    notes = []
    for file in glob.glob(f"{data_path}/*.mid"):
        midi = converter.parse(file)
        print(f"Parsing {file}")
        try:
            s2 = instrument.partitionByInstrument(midi)
            parsing_notes = s2.parts[0].recurse()
        except:
            parsing_notes = midi.flat.notes

        for element in parsing_notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('./notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, sequence_length):
    name_of_pitch = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(name_of_pitch)}
    inputs, outputs = [], []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        inputs.append([note_to_int[char] for char in sequence_in])
        outputs.append(note_to_int[sequence_out])

    n_vocab = len(name_of_pitch)
    inputs = np.reshape(inputs, (len(inputs), sequence_length, 1)) / n_vocab
    outputs = np.array(outputs)
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.long), name_of_pitch, n_vocab
