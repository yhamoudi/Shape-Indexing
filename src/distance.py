import argparse
from eigenvalues import compute_descriptor, compute_eigenvalues
from multiprocessing import Pool
import image
import scipy.spatial.distance


def descriptor(file):
    im = image.Image(file)
    im.normalize(50)
    im_eigenvalues = compute_eigenvalues(im.image)
    return compute_descriptor(im_eigenvalues)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the distance between two images')
    parser.add_argument('file1', metavar='F', help='the first image')
    parser.add_argument('file2', metavar='F', help='the second image')

    args = parser.parse_args()

    pool = Pool(2)
    r = pool.map(descriptor,[args.file1, args.file2])

    v1 = r[0]
    v2 = r[1]

    distance = scipy.spatial.distance.cosine(v1, v2)

    print(distance)