import argparse
import pickle
import laplacian

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file where are stored the eigenvalues')
    args = parser.parse_args()

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))

    for name in eigenvalues:
        l = eigenvalues[name]


