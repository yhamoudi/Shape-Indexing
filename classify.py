import argparse
import pickle
from eigenvalues import compute_descriptor
import database
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file where are stored the eigenvalues')


    args = parser.parse_args()

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))

    data_set = database.DataSet(0.80)

    for name in eigenvalues:
            ev_list = eigenvalues[name]
            category = name.split('-')[0]
            data_set.add(category, compute_descriptor(ev_list))

    train_set = data_set.get_train_set()
    test_set = data_set.get_test_set()
    vector_to_classify = test_set[0, 0:train_set.shape[1]-1]
    classifier = database.EuclideanClassifier(train_set)

    probability_vector = classifier.probabilistic_classification(vector_to_classify)


    id_classe = np.argmax(probability_vector)
    classes_name = {}
    for name in data_set.classes:
        classes_name[data_set.classes[name]] = name

    for i in range(0, probability_vector.shape[0]):
        print('%s: %.4f' %(classes_name[i+1], probability_vector[i]))
