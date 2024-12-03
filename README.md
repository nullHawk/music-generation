# Music Generator

A Python-based AI application that generates music using an LSTM-based RNN model trained on music data. This project provides an end-to-end solution to compose music and play it as audio, leveraging the power of deep learning and Gradio for an interactive user experience.

## Features
- **Music Generation**: Uses a trained RNN model to generate music in ABC notation.
- **Interactive Interface**: Gradio-based interface for easy interaction.
- **LSTM/GRU Support**: Configurable choice of LSTM or GRU for music generation.
- **Audio Conversion**: Converts ABC notation into playable audio.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Required Python packages (install via `requirements.txt`):
  - `torch`
  - `gradio`
  - `numpy`
  - `abc2midi`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/nullHawk/music-generation
   cd music-generation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the trained model checkpoint at `checkpoint/model.pth`.

---

## Usage

### Running the Application
Launch the Gradio interface:
```bash
python app.py
```

### Interacting with the App
- Open the Gradio interface in your browser.
- Click the "Generate Music" button to compose a new song.
- Listen to the generated audio directly within the interface.

---

## Example Output

1. Click the **Generate Music** button.
2. The app generates an ABC-notation music sequence using the RNN model.
3. The ABC notation is converted to playable audio.
4. Listen to the generated music in the interface.

---