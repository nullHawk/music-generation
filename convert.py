from music21 import converter, stream
from midi2audio import FluidSynth
import subprocess

def abc_to_audio(abc_notation, output_format='wav',sound_font="FluidR3_GM.sf2"):
    """ Convert ABC notation to wav file. """
    abc_file = 'output.abc'
    with open(abc_file, 'w') as f:
        f.write(abc_notation)
    subprocess.run(['abc2midi', abc_file, '-o', "output.midi"])
    fs = FluidSynth()
    fs.midi_to_audio("output.midi", "output.wav")
    return "output.wav"


if __name__ == '__main__':
    abc_to_audio("""X:12
T:Byrne: Triop
C:Trad Figne
Z:id:hn-hornpipe-53
M:C|
K:G
(3DFB d2dc | def2 edef | e2a2 df | g4- gdBG | A4G | A4 :|
|: ae edc | edcB A2B2 | A2G2 | G6 d2 | e4^c4 | d4 d4 | ed e2 | d4 ||
P:variations:
|: ABA AGE|F2A d2A|d2g d2:|
a2f fef aba|a2f g2e fed|c2A GBd|f2g g2a|bgb aag|dcB B2G|A2G A2G:|
|:F2A A2G|AGE G2d||
P:variations
|: AGF GBd | cde d2B | c2c c2A :|
|: de fe | fdfe dFAd | A2AG A2f2 | g2ag e2B2 | A2AB ^cdce | d2d>c | B4z2 | B4 | A4G2 | ^F4G4 | G4 :|
|: G^F G2 | c4 ||
GBdB | c2 ded2 | c2B2c2 | d2c2B2 | c2d2 | c2B2 | A4 :|""")