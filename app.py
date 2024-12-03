import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from model import MusicLSTM
from train import DataLoader, Config, generate_song as generate_ABC_notation
from utils import load_vocab
from convert import abc_to_audio

class GradioApp():
    def __init__(self):
        # Set up configuration and data
        self.config = Config()
        self.CHECKPOINT_FILE = "checkpoint/model.pth"
        self.data_loader = DataLoader(self.config.INPUT_FILE, self.config)
        self.checkpoint = torch.load(self.CHECKPOINT_FILE, weights_only=False)
        char_idx, char_list = load_vocab()
        self.model = MusicLSTM(
            input_size=len(char_idx),
            hidden_size=self.config.HIDDEN_SIZE,
            output_size=len(char_idx),
        )
        self.model.load_state_dict(self.checkpoint)
        self.model.eval()

        #Setup Interface
        # self.input = gr.Button("Generate Music")
        # self.output = gr.Audio(label="Generated Music")
        # # self.output = gr.Textbox("")
        # self.interface = gr.Interface(
        #     fn=self.generate_music,
        #     inputs=self.input,
        #     outputs=self.output,
        #     title="AI Music Generator",
        #     description="Generate a new song using a trained RNN model."
        # )
    def launch(self):
        # Define Gradio interface without a clear button
        with gr.Blocks() as demo:
            gr.Markdown("# AI Music Generator")
            gr.Markdown("Click the button below to generate a new random song using a trained RNN model.")
            
            generate_button = gr.Button("Generate Music")
            output_audio = gr.Audio(label="Generated Music")
            
            generate_button.click(self.generate_music, inputs=None, outputs=output_audio)
        
        demo.launch()
    # def launch(self):
    #     self.interface.launch()

    def generate_music(self, input):
        """Generate a new song using the trained model."""
        abc_notation = generate_ABC_notation(self.model, self.data_loader)
        abc_notation = abc_notation.strip("<start>").strip("<end>").strip()
        audio = abc_to_audio(abc_notation)
        return audio

if __name__ == '__main__':
    app = GradioApp()
    app.launch()

    