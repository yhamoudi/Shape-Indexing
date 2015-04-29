from __future__ import division
import math
import argparse
from pyaudio import PyAudio

import laplacian
from image import Image
import numpy as np
import os.path
import pickle


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


def compute_eigenvalues(img_file):
    im = Image(img_file)
    im.normalize()
    return laplacian.compute_eigenvalues(im.image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play an image')
    parser.add_argument('file', help='the image')
    parser.add_argument('--eigenvalues', help='the precomputed eigenvalues')
    parser.set_defaults(eigenvalues=1)
    args = parser.parse_args()

    if os.path.isfile(args.eigenvalues):
        print(args.file.split('/')[-1])
        eigenvalues_list = pickle.load(open(args.eigenvalues, "rb"))
        if args.file.split('/')[-1] in eigenvalues_list:
            eigenvalues = eigenvalues_list[args.file.split('/')[-1]]
        else:
            eigenvalues = compute_eigenvalues(args.file)
    else:
        eigenvalues = compute_eigenvalues(args.file)

    eigenvalues = np.array(eigenvalues, dtype=float)
    frequencies = np.array(40 * np.sqrt(eigenvalues[0:3]), dtype=int)
    print(frequencies)
    sine_tone(
        # see http://www.phy.mtu.edu/~suits/notefreqs.html
        frequencies=frequencies.tolist(), # Hz, waves per second A4
        amplitudes=np.ones(3).tolist(),
        duration=2., # seconds to play sound
        volume=.3, # 0..1 how loud it is
        # see http://en.wikipedia.org/wiki/Bit_rate#Audio
        sample_rate=22050 # number of samples per second
)
