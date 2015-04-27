import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file were are stored the eigenvalues')
    args = parser.parse_args()

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))
    print(eigenvalues.keys())
