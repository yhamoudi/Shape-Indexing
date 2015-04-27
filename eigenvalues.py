#!/usr/bin/env python3

import re # regular expressions
import argparse
import pickle

import numpy as np
import scipy.misc
from matplotlib import pyplot

import laplacian
from multiprocessing import Pool



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
        # boolean matrix : False if black, True if white
        # (the picture itself is white) :
        self.image = np.vectorize(lambda x: x != 0)(image)

    def height(self):
        return self.image.shape[0]

    def weight(self):
        return self.image.shape[1]

    def make_uniform(self):
        self.crop()
        self.resize()
        self.add_black_edges()

    def resize(self, max_edge_size=50):
        alpha = float(max_edge_size) / float(max(self.height(), self.weight()))
        self.image = scipy.misc.imresize(self.image, alpha)
        f = np.vectorize(lambda x: x > 0.5)
        self.image = f(self.image)

    def __remove_first_line(self):
        if self.image[0,:].sum() == 0:
            self.image = self.image[1:, :]
            return self.__remove_first_line()

    def __remove_last_line(self):
        if self.image[self.height()-1,:].sum() == 0:
            self.image = self.image[0:self.height()-1, :]
            return self.__remove_last_line()

    def __remove_first_column(self):
        if self.image[:,0].sum() == 0:
            self.image = self.image[:, 1:]
            return self.__remove_first_column()

    def __remove_last_column(self):
        if self.image[:, self.weight()-1].sum() == 0:
            self.image = self.image[:, 0:self.weight()-1]
            return self.__remove_last_column()

    def crop(self): # crop the image until all borders contain a white pixel
        self.__remove_last_column()
        self.__remove_first_column()
        self.__remove_first_line()
        self.__remove_last_line()

    def add_black_edges(self):  # add a black border all around the image
        im = np.zeros((self.height()+2, self.weight()+2), dtype=np.int)
        im[1:self.height()+1, 1:self.weight()+1] = self.image
        self.image = im

    def print(self):
        pyplot.imshow(self.image, pyplot.cm.gray)
        pyplot.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute eigenvalues of an image')

    parser.add_argument('files', metavar='F', nargs='+', help='the pgm image you want to compute the eigenvalues')
    parser.add_argument('--output', help='output file', required=True)
    parser.add_argument('--ncpus', help='number of cpus', type=int)

    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.set_defaults(show=False, ncpus=1)

    args = parser.parse_args()

    def compute_eigenvalues(path_image):
        try:
            name = path_image.split('/')[-1].split('.')[0]
            im = Image(path_image)
            im.make_uniform()
            if args.show:
                im.print()

            eigenvalues = laplacian.compute_eigenvalues(im.image)
            print(name + ': ' + str(laplacian.compute_descriptor(eigenvalues)))
            return name, eigenvalues
        except ValueError:
            print('Warning: a problem occurs with ' + path_image)


    pool = Pool(args.ncpus)
    results = {}
    for path_image in args.files:
        name = path_image.split('/')[-1].split('.')[0]
        results[name] = pool.apply_async(compute_eigenvalues, [path_image])

    output = {}
    for name in results.keys():
        output[name] = results[name].get()

    pickle.dump(output, open(args.output, "wb"))