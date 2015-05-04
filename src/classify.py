import argparse
import pickle
from eigenvalues import compute_descriptor, compute_eigenvalues
import database
import numpy as np
import image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('--eigenvalues', metavar='F', help='the file where are stored the eigenvalues')
    parser.add_argument('--classes', metavar='F', help='the file where are stored the classes')
    parser.add_argument('file', metavar='F', help='the file to classify')

    args = parser.parse_args()

    classes = database.load_classes(args.classes)
    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))

    im = image.Image(args.file)
    im.normalize(50)
    im_eigenvalues = compute_eigenvalues(im.image)

    im_descriptor = compute_descriptor(im_eigenvalues)

    data_set = database.DataSet(1.0, classes)

    for name in eigenvalues:
        ev_list = eigenvalues[name]
        category = name.split('-')[0]
        data_set.add(category, compute_descriptor(ev_list))

    train_set = data_set.get_train_set()

    vector_to_classify = data_set.normalize_vector(im_descriptor)

    classifier = database.DistanceClassifier(train_set)

    probability_vector = classifier.probabilistic_classification(vector_to_classify)

    classes_name = {}
    for name in data_set.classes:
        classes_name[data_set.classes[name]] = name

    for i in range(0, probability_vector.shape[0]):
        print('%.4f' % (probability_vector[i]))