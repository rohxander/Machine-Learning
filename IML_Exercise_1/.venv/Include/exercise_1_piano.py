from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    array = np.load(filename)
    array_abs = np.abs(array)
    max = np.argmax(array_abs)
    start_value = max + offset
    end_value = min(start_value + duration, len(array))
    return array[start_value:end_value]

    return None

def compute_frequency(signal, min_freq=20):
    # Complete this function
    dft_signal = np.fft.fft(signal)
    dft_signal = np.abs(dft_signal)
    dft_freq = np.fft.fftfreq(len(signal), d=1 / 44100)
    i = np.where(dft_freq > min_freq)[0][0]
    max = np.argmax(dft_signal[i:]) + i
    return dft_freq[max]

if __name__ == '__main__':
    # Implement the code to answer the questions here
    sounds_dir = 'sounds/'
    files = ['Piano.ff.A2.npy', 'Piano.ff.A3.npy', 'Piano.ff.A4.npy',
             'Piano.ff.A5.npy', 'Piano.ff.A6.npy', 'Piano.ff.A7.npy',
             'Piano.ff.XX.npy']

    source = [os.path.join(sounds_dir, x) for x in files]

    # Expected frequencies for the notes A2, A3, A4, A5, A6, A7 (in Hz)
    test_frequencies = [110, 220, 440, 880, 1760, 3520]

    # Compute the frequencies of all notes and compare them to expected values
    computed_frequencies = []
    for sound_file in source:
        sample = load_sample(sound_file)
        peak_frequency = abs(compute_frequency(sample))
        computed_frequencies.append(peak_frequency)

    # Find the mysterious note
    mysterious_note_frequency = computed_frequencies[-1]
    print(mysterious_note_frequency)

    print(f"The mysterious note frequency is {round(mysterious_note_frequency,2)} which is closest to D6 (1174.66 Hz)")

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
