#!/usr/bin/env python3

import re # regular expressions
import numpy as np
import sys
from matplotlib import pyplot

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

    def print(self):
      pyplot.imshow(self.image, pyplot.cm.gray)
      pyplot.show()


if __name__ == "__main__":
    im = Image(sys.argv[1])
    im.print()
