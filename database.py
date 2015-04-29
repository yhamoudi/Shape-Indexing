import argparse
import pickle
import laplacian
import numpy as np
import random

from sklearn.lda import LDA
from sklearn.externals import joblib


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


class LinearDiscriminantAnalysis(object):
    def __init__(self, input_matrix, labels):
        self.x = input_matrix
        self.y = labels
        self.clf = LDA()

    def train(self):
        self.clf.fit(self.x, self.y)

    def predict(self, x):
        return self.clf.predict(x)

    def save_model(self, file):
        joblib.dump(self.clf, file)


class Classifier:
    def __init__(self, train_set):
        descriptor_size = train_set.shape[1]-1
        input_matrix = train_set[:, 0:descriptor_size-1]
        labels = train_set[:, descriptor_size].astype(int)
        self.__classifier = LinearDiscriminantAnalysis(input_matrix=input_matrix,
                                                       labels=labels)

    def train(self):
        self.__classifier.train()

    def evaluation(self, test_set):
        descriptor_size = test_set.shape[1]-1
        input_matrix = test_set[:, 0:descriptor_size-1]
        labels = test_set[:, descriptor_size].astype(int)
        print(labels)
        estimated_answers = self.__classifier.predict(input_matrix)

        correct_answers = np.sum(estimated_answers == labels)
        ratio_correct_answers = float(correct_answers)/float(labels.shape[0])

        return ratio_correct_answers



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file where are stored the eigenvalues')
    args = parser.parse_args()

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))

    data_set = DataSet(0.95)

    for name in eigenvalues:
        ev_list = eigenvalues[name]
        category = name.split('-')[0]
        data_set.add(category, laplacian.compute_descriptor(ev_list))

    classifier = Classifier(data_set.get_train_set())
    classifier.train()
    print(classifier.evaluation(data_set.get_test_set()))