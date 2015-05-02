#!/usr/bin/env python3

from __future__ import division
import math
import argparse
from pyaudio import PyAudio
import eigenvalues
from image import Image
import numpy as np
import os.path
import pickle
import random
import time

# Using http://stackoverflow.com/questions/974071/python-library-for-playing-fixed-frequency-sound
def sine_tone(frequencies, amplitudes, duration, volume=1.0, sample_rate=22050):
    n_samples = int(sample_rate * duration)
    restframes = n_samples % sample_rate

    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(1), # 8bit
                    channels=1, # mono
                    rate=sample_rate,
                    output=True)

    def s(t):
        r = 0
        for i in range(0, len(frequencies)):
            r += volume * amplitudes[i] * math.sin(2 * math.pi * frequencies[i] * t / sample_rate)
        return r

    samples = (int(s(t) * 0x7f + 0x80) for t in range(n_samples))
    for buf in zip(*[samples]*sample_rate): # write several samples at a time
        stream.write(bytes(bytearray(buf)))

    # fill remainder of frameset with silence
    stream.write(b'\x80' * restframes)

    stream.stop_stream()
    stream.close()
    p.terminate()

def produce_sound(eigenvalues): # produce a sound from eigenvalues
    eigenvalues = np.array(eigenvalues, dtype=float)
    frequencies = np.array(40 * np.sqrt(eigenvalues[0:3]), dtype=int)
    print(frequencies)
    sine_tone(
        frequencies=frequencies.tolist(), # Hz, waves per second A4
        amplitudes=np.ones(3).tolist(),
        duration=2., # seconds to play sound
        volume=.3, # 0..1 how loud it is
        sample_rate=22050 # number of samples per second
    )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play an image')
    parser.add_argument('file', help='Name of the image')
    parser.add_argument('--ev', help='file containing the eigenvalues')
    parser.set_defaults(ev='eigenvalues/eigenvalues.db')
    args = parser.parse_args()

    eigenvalues_list = pickle.load(open(args.ev, "rb")) # eigenvalues already computed
    name = args.file.split('/')[-1]
    if name in eigenvalues_list:
        eigenvalues = eigenvalues_list[name]
    else:
        im = Image(args.file)
        im.normalize()
        eigenvalues = eigenvalues.compute_eigenvalues(im.image)

    produce_sound(eigenvalues) # produce the sound associated to the eigenvalues of the input image
