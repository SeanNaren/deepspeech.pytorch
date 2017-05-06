from tempfile import NamedTemporaryFile
import numpy as np
from scipy.io.wavfile import read as wav_read
import os


def augment_audio_with_sox(path, sample_rate, tempo=1.0, gain=0.0):
    """
    Changes tempo and gain of the input file with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox {} -r {} -c 1 -b 16 {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                            augmented_filename, " ".join(sox_augment_params))
        os.system(sox_params)
        sr, y = wav_read(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
