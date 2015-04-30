#!/usr/bin/env python3

from __future__ import division
import math
import argparse
from pyaudio import PyAudio

import laplacian
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
        # see http://www.phy.mtu.edu/~suits/notefreqs.html
        frequencies=frequencies.tolist(), # Hz, waves per second A4
        amplitudes=np.ones(3).tolist(),
        duration=2., # seconds to play sound
        volume=.3, # 0..1 how loud it is
        # see http://en.wikipedia.org/wiki/Bit_rate#Audio
        sample_rate=22050 # number of samples per second
    )

def sound_game():
    eigenvalues = pickle.load(open('eigenvalues/eigenvalues_arranged.db', "rb"))
    dif = int(input('Enter the difficulty (2-10): ')) # number of sounds to heard
    category = random.choice(list(eigenvalues.keys())) # a random category
    sample = random.randint(0,len(eigenvalues[category])-1) # an object in this category
    print("Listen this sample of " + category)
    time.sleep(2)
    produce_sound(eigenvalues[category][sample][1])
    print("Now listen these " + str(dif) + " sounds:")
    now = random.randint(0,dif-1) # time to listen an object of the same category than sample
    for i in range(0,dif):
      time.sleep(2)
      print("Sound " + str(1+i))
      if i == now:
        produce_sound(eigenvalues[category][random.randint(0,len(eigenvalues[category])-1)][1])
      else:
        rand_cat = random.choice(list(eigenvalues.keys()))
        while rand_cat == category:
          rand_cat = random.choice(eigenvalues.keys())
        produce_sound(eigenvalues[rand_cat][random.randint(0,len(eigenvalues[rand_cat])-1)][1])
    answer = int(input("Which one of these sounds was %s?\n" % category))
    if answer-1 == now:
        print("Success")
    else:
        print("Fail, it was number " + str(now+1))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play an image')
    parser.add_argument('file', help='Name of the image')
    args = parser.parse_args()

    eigenvalues_list = pickle.load(open('eigenvalues/eigenvalues.db', "rb")) # eigenvalues already computed
    name = args.file.split('/')[-1]
    if name in eigenvalues_list:
        eigenvalues = eigenvalues_list[name]
    else:
        im = Image(args.file)
        im.normalize()
        eigenvalues = laplacian.compute_eigenvalues(im.image)

    produce_sound(eigenvalues) # produce the sound associated to the eigenvalues of the input image
