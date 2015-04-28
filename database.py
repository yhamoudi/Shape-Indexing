import argparse
import pickle
import laplacian
import numpy as np
import random


class DataSet:
    __test_set = []
    __train_set = []

    __classes = {}

    def __init__(self, ratio_train):
        self.__ratio_train = ratio_train

    def add(self, category_name, descriptor):
        if not(category_name in self.__classes):
            self.__classes[category_name] = len(self.__classes) + 1
        id_category = self.__classes[category_name]
        entry = np.array(descriptor + [float(id_category)], dtype=float)
        if random.random() < self.__ratio_train:
            self.__train_set.append([entry])
        else:
            self.__test_set.append([entry])

    def get_train_set(self):
        return np.concatenate(self.__train_set)

    def get_test_set(self):
        return np.concatenate(self.__test_set)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file where are stored the eigenvalues')
    args = parser.parse_args()

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))

    data_set = DataSet(0.8)

    for name in eigenvalues:
        ev_list = eigenvalues[name]
        category_name = name.split('-')[0]
        data_set.add(category_name, laplacian.compute_descriptor(ev_list))

    print(data_set.get_train_set().shape)
    print(data_set.get_train_set())
