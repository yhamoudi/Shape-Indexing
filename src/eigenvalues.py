#!/usr/bin/env python3

import sys
import argparse
import pickle
from multiprocessing import Pool
import os.path
import scipy.misc
from image import Image
import numpy as np

# Generate the following matrix:
# [[-A.  In   0.   0.  0.]
# [ In  -A.  In    0.  0.]
# [ 0.  In   -A   In   0.]
# [ 0.  0.   In   -A.  In]
# [ 0.  0.    0.  In  -A.]]

# With A the following n*n matrix:

# [[-4.  1.  0.  0.]
#  [ 1. -4.  1.  0.]
#  [ 0.  1. -4.  1.]
#  [ 0.  0.  1. -4.]]

def generate_laplacian_matrix(height, width):
    N = height*width
    a = np.diagflat(-4*np.ones(N), k=0)
    b = np.diagflat(np.ones(N-1), k=1)
    c = np.diagflat(np.ones(N-1), k=-1)
    d = np.diagflat(np.ones(N-width), k=-width)
    e = np.diagflat(np.ones(N-width), k=width)
    return (a+b+c+d+e)*(height*width)

# Given a n*n image (numpy matrix), compute the corresponding vector
# Then compute the eigenvalues of the image v
def compute_eigenvalues(img):
    height = img.shape[0]
    width = img.shape[1]
    img_flat = np.empty_like(img)
    img_flat[:] = img
    img_flat.resize(height*width,1)
    M = generate_laplacian_matrix(height, width)
    laplacian = M * img_flat
    w= np.linalg.eigvals(laplacian).real
    w = w[w<-0.001]
    return sorted(-w)

def arrange_eigenvalues(eigenvalues): # produce a structured database
    output = {}
    for image in eigenvalues:
      [category,number] = image.split('-')
      if category in output:
        output[category] = output[category] + [[number,eigenvalues[image]]]
      else:
        output[category] = [[number,eigenvalues[image]]]
    return output

def compute_descriptor(eigenvalues):
    descriptor = []
    for i in range(1, 30):
        descriptor.append(eigenvalues[0]/eigenvalues[i])

    #for i in range(1, 10):
    #    descriptor.append(eigenvalues[i]/eigenvalues[i+1])

    return descriptor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute eigenvalues of an image')
    parser.add_argument('dir', help='the directory containing pgm images you want to compute the eigenvalues')
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
            im.normalize(50)
            print(name + ' is processed')
            return (name, compute_eigenvalues(im.image))

    pool = Pool(args.ncpus)
    output.update(dict(pool.map(parallel_compute_eigenvalues,os.listdir(args.dir))))

    pickle.dump(output, open(args.output, "wb"))
