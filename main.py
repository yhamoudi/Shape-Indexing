#!/usr/bin/env python3

import re # regular expressions
import numpy as np
import scipy.misc
import sys
from matplotlib import pyplot
import laplacian
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

    def resize(self, alpha): # resize the image by a factor alpha
        self.image = scipy.misc.imresize(self.image, alpha)
        f = np.vectorize(lambda x: x > 0.5)
        self.image = f(self.image)
        self.height = self.image.shape[0]
        self.weight = self.image.shape[1]

    def remove_first_line(self):
        if self.image[0,:].sum() == 0:
            self.image = self.image[1:, :]
            self.height -= 1
            return self.remove_first_line()

    def remove_last_line(self):
        if self.image[self.height-1,:].sum() == 0:
            self.image = self.image[0:self.height-1, :]
            self.height -= 1
            return self.remove_last_line()

    def remove_first_column(self):
        if self.image[:,0].sum() == 0:
            self.image = self.image[:, 1:]
            self.weight -= 1
            return self.remove_first_column()

    def remove_last_column(self):
        if self.image[:, self.weight-1].sum() == 0:
            self.image = self.image[:, 0:self.weight-1]
            self.weight -= 1
            return self.remove_last_column()

    def crop(self): # crop the image until all borders contain a white pixel
        self.remove_last_column()
        self.remove_first_column()
        self.remove_first_line()
        self.remove_last_line()

    def add_black_edges(self): # add a black border all around the image
        im = np.zeros((self.height+2, self.weight+2), dtype=np.int)
        im[1:self.height+1, 1:self.weight+1] = self.image
        self.image = im
        self.height += 2
        self.weight += 2

    def print(self):
      pyplot.imshow(self.image, pyplot.cm.gray)
      pyplot.show()

if __name__ == "__main__":
    im = Image(sys.argv[1])
    im.crop()
    im.resize(0.25)
    im.add_black_edges()
    #im.print()

    eigenvalues = laplacian.compute_eigenvalues(im.image)
    print(eigenvalues)
    #print(laplacian.compute_descriptor(eigenvalues))
