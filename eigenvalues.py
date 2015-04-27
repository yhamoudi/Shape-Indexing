#!/usr/bin/env python3

import sys
import re # regular expressions
import argparse
import pickle
from multiprocessing import Pool
import os.path
import numpy as np
import scipy.misc

import laplacian
from image import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute eigenvalues of an image')
    parser.add_argument('files', metavar='F', nargs='+', help='the pgm image you want to compute the eigenvalues')
    parser.add_argument('--output', help='output file', required=True)
    parser.add_argument('--ncpus', help='number of cpus', type=int)
    parser.set_defaults(ncpus=1,)
    args = parser.parse_args()

    output = {}
    if os.path.isfile(args.output):
        output = pickle.load(open(args.output, "rb"))

    def parallel_compute_eigenvalues(path_image):
        try:
            im = Image(path_image)
            im.normalize()
            eigenvalues = laplacian.compute_eigenvalues(im.image)
            print(path_image)
            return eigenvalues
        except ValueError:
            print('Warning: a problem occurs with ' + path_image)

    pool = Pool(args.ncpus)
    for path_image in args.files:
        name = path_image.split('/')[-1].split('.')[0]
        if not name in output:
            output[name] = pool.apply_async(parallel_compute_eigenvalues, [path_image]).get()

    pickle.dump(output, open(args.output, "wb"))