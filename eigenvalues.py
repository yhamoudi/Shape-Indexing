#!/usr/bin/env python3

import sys
import argparse
import pickle
from multiprocessing import Pool
import os.path
import scipy.misc
import laplacian
from image import Image

def arrange_eigenvalues(eigenvalues): # produce a structured database
    output = {}
    for image in eigenvalues:
      [category,number] = image.split('-')
      if category in output:
        output[category] = output[category] + [[number,eigenvalues[image]]]
      else:
        output[category] = [[number,eigenvalues[image]]]
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute eigenvalues of an image')
    parser.add_argument('dir', help='the directory of the pgm images you want to compute the eigenvalues')
    parser.add_argument('--output', help='output file', required=True)
    parser.add_argument('--ncpus', help='number of cpus', type=int)

    parser.set_defaults(ncpus=1)
    args = parser.parse_args()

    output = {}
    if os.path.isfile(args.output):
        output = pickle.load(open(args.output, "rb"))

    def parallel_compute_eigenvalues(name):
        if not name in output:
            im = Image(args.dir+'/'+name)
            im.normalize()
            print(name + ' is processed')
            return (name, laplacian.compute_eigenvalues(im.image))

    pool = Pool(args.ncpus)
    output.update(dict(pool.map(parallel_compute_eigenvalues,os.listdir(args.dir))))

    pickle.dump(output, open(args.output, "wb"))
