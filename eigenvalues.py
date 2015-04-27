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
    #im = Image(sys.argv[1])
    #im.normalizem()
    #im.print()

    parser = argparse.ArgumentParser(description='Compute eigenvalues of an image')

    parser.add_argument('files', metavar='F', nargs='+', help='the pgm image you want to compute the eigenvalues')
    parser.add_argument('--output', help='output file', required=True)
    parser.add_argument('--ncpus', help='number of cpus', type=int)

    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.set_defaults(show=False, ncpus=1)

    args = parser.parse_args()

    output = {}
    if os.path.isfile(args.output):
        output = pickle.load(open(args.output, "rb" ))

    def compute_eigenvalues(path_image):
        try:
            name = path_image.split('/')[-1].split('.')[0]
            if not name in output:
                im = Image(path_image)
                im.normalize()
                if args.show:
                    im.print()

                eigenvalues = laplacian.compute_eigenvalues(im.image)
            else:
                eigenvalues = output[name]

            print(name)

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
