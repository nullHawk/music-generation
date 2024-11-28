import os
import subprocess

def abc_to_wav(abc_content, output_wav):
    """
    Convert ABC notation to WAV file.

    Parameters:
        abc_content (str): ABC notation as a string.
        output_wav (str): Path to save the output WAV file.
    """
    # Step 1: Save ABC content to a temporary file
    abc_file = "temp_music.abc"
    midi_file = "temp_music.mid"
    try:
        with open(abc_file, "w") as f:
            f.write(abc_content)

        # Step 2: Convert ABC to MIDI using abc2midi
        subprocess.run(["abc2midi", abc_file, "-o", midi_file], check=True)

        # Step 3: Convert MIDI to WAV using ffmpeg
        subprocess.run(["ffmpeg", "-y", "-i", midi_file, output_wav], check=True)
        print(f"WAV file created: {output_wav}")

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(abc_file):
            os.remove(abc_file)
        if os.path.exists(midi_file):
            os.remove(midi_file)

# Example ABC content
abc_content = """
X:23
T:Walsh's of Marro Keveran
Z:<ericain
Z:id:hn-hornpipe
Z:B/c/d|1
e>d B>A|BB BA|BA A>B|BA Ad|B>A GE|FB FD|AF FA|de/d/ cf|eA ce|f>e d>c|BA B/c/A/G/|FA AF|A2 A2:|
P:Variations:
|:~G3d ~B3G|AGFG EFGA|FA~A2 B2AG|FAdF DFAF|
GABG ABdB|cBAG ABcd|
efge ~a3b|afaf gfed|BGBd gdBd|
ggba Bceg|af~e2 d2ef|gbga bgef|gfed B2AB|AG~F2 G2:|
P:variations one wure observation galouvielle@free.fr
M:2/4
L:1/4
Q:3/8=16
K:G
GD|G>G BG|EG ED|EA cA|GA Bd|ea ga/g/|ed dB|A2 BG|A>B ce|ag/e/ g>a|bg bg|fd d>c|Bc AB|c2 c2:|
|:e a2|ag a2 | b2a bg|fe f/e/d|cA FA|1 B4:|
|:g>f ed|B/A/B/A/ G2:|
|:gf ge|
e/a/ g/a/g|ed d>e||
|:a>g g/f/e/d/|Be e/d/e/d/|Be ef|ge de|af e>d|cA/A/ B/c/A/G/|
FA A>c|eB/A/ BA|F>A B/c/d|BA B/c/B:|2 BA d>B|AF A2:|
"""

# Convert the ABC content to WAV
abc_to_wav(abc_content, "output_music.wav")
