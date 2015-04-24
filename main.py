#!/usr/bin/env python3

import re # regular expressions
import numpy as np
import scipy.misc
import sys
from matplotlib import pyplot
import math

class Image:
    def __init__(self, filename):
      with open(filename, 'rb') as f: #  From http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
        buffer = f.read()
      try:
        header, width, height, maxval = re.search(
          b"(^P5\s(?:\s*#.*[\r\n])*"
          b"(\d+)\s(?:\s*#.*[\r\n])*"
          b"(\d+)\s(?:\s*#.*[\r\n])*"
          b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
      except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
      image = np.frombuffer(buffer,
                           dtype='u1',
                           count=int(width)*int(height),
                           offset=len(header)
                           ).reshape((int(height), int(width)))
      self.image = np.vectorize(lambda x: x != 0)(image) # boolean matrix : False if black, True if white (the picture itself is white)
      self.height = int(height)
      self.weight = int(width)


    def resize(self, alpha):
        # print(self.image.shape)
        # n = self.image.shape[0]
        # m = self.image.shape[1]
        # new_n = int(math.ceil(n/float(h_a)))
        # new_m = int(math.ceil(m/float(w_a)))
        #
        # new_image0 = np.concatenate((self.image, np.zeros((new_n*h_a-n, m))), axis=0)
        # new_im = new_image0.reshape((h_a, new_n*m), order='A').mean(axis=0).reshape((new_n, m))
        #
        # print(new_im.shape)
        #
        # pyplot.imshow(new_im, pyplot.cm.gray)
        # pyplot.show()

        self.image = scipy.misc.imresize(self.image, alpha)
        f = np.vectorize(lambda x: x > 0.5)
        self.image = f(self.image)
        self.height = self.image.shape[0]
        self.weight = self.image.shape[1]



    def print(self):
      pyplot.imshow(self.image, pyplot.cm.gray)
      pyplot.show()


if __name__ == "__main__":
    im = Image(sys.argv[1])
    #im.print()
    im.resize(0.7)
    im.print()