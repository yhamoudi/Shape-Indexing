#!/usr/bin/env python3

import argparse
import pickle
from eigenvalues import compute_descriptor
import numpy as np
import random
import scipy.spatial.distance
from multiprocessing import Pool


class DataSet:
    def __init__(self, ratio_train, classes):
        self.ratio_train = ratio_train
        self.test_set = []
        self.train_set = []
        self.classes = classes
        self.mean = None
        self.std = None

    def add(self, category_name, descriptor):
        id_category = self.classes[category_name]
        entry = np.array(descriptor + [float(id_category)], dtype=float)
        if random.random() < self.ratio_train:
            self.train_set.append([entry])
        else:
            self.test_set.append([entry])

    def get_train_set(self):
        matrix = np.concatenate(self.train_set)
        self.mean = matrix[:, 0:matrix.shape[1]-1].mean(axis=0)
        self.std = matrix[:, 0:matrix.shape[1]-1].std(axis=0)
        matrix[:, 0:matrix.shape[1]-1] = (matrix[:, 0:matrix.shape[1]-1] - self.mean) / self.std
        return matrix

    def get_test_set(self):
        matrix = np.concatenate(self.test_set)
        matrix[:, 0:matrix.shape[1]-1] = (matrix[:, 0:matrix.shape[1]-1] - self.mean) / self.std
        return matrix

    def normalize_vector(self, vector):
        return (vector - self.mean) / self.std


class DistanceClassifier:
    def __init__(self, train_set):
        descriptor_size = train_set.shape[1]-1
        self.input_matrix = train_set[:, 0:descriptor_size]
        self.labels = train_set[:, descriptor_size].astype(int)
        self.distance = self.cosine_distance # distance to use

    @staticmethod
    def euclidean_distance(a, b):
        return scipy.spatial.distance.euclidean(a, b)

    @staticmethod
    def minkowski_distance(a, b):
        return scipy.spatial.distance.minkowski(a, b, 10)

    @staticmethod
    def sqeuclidean_distance(a, b):
        return scipy.spatial.distance.sqeuclidean(a, b)

    @staticmethod
    def cosine_distance(a, b):
        return scipy.spatial.distance.cosine(a, b)

    def classify(self, normalized_descriptor):
        f = lambda v: self.distance(v, normalized_descriptor)

        result = np.fromiter(map(f, self.input_matrix),
                             dtype=self.input_matrix.dtype, count=self.input_matrix.shape[0])

        return self.labels[np.argmin(result)]

    def probabilistic_classification(self, normalized_descriptor):
        f = lambda v: self.distance(v, normalized_descriptor)
        distances = np.fromiter(map(f, self.input_matrix),
                             dtype=self.input_matrix.dtype, count=self.input_matrix.shape[0])

        list_classes = np.unique(self.labels)
        nb_classes = list_classes.shape[0]
        classes_dst = np.zeros(nb_classes)
        for i in range(0, nb_classes):
            di = np.min(distances[np.where(self.labels == i+1)])
            if di < 0.00001:
                classes_dst = np.zeros(nb_classes)
                classes_dst[i] = 1
                return classes_dst
            else:
                classes_dst[i] = 1.0 / np.min(distances[np.where(self.labels == i+1)])

        classes_dst = classes_dst / np.sum(classes_dst)

        return classes_dst


    def evaluation(self, test_set):
        descriptor_size = test_set.shape[1]-1
        input_matrix = test_set[:, 0:descriptor_size]
        labels = test_set[:, descriptor_size].astype(int)

        estimated_answers = np.fromiter(map(self.classify, input_matrix), dtype=int, count=input_matrix.shape[0])

        correct_answers = np.sum(estimated_answers == labels)
        return float(correct_answers)/float(labels.shape[0])


def load_classes(file_classes):
    with open(file_classes, 'r') as csv_file:
        classes = {}
        i = 1
        for line in csv_file:
            classes[line.split(',')[0]] = i
            i += 1
        return classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file where are stored the eigenvalues')
    parser.add_argument('--classes', metavar='F', help='the file where are stored the classes')
    parser.add_argument('--niters', help='number of iterations (to reduce the std)', type=int)
    parser.add_argument('--LDA', dest='lda', action='store_true')

    parser.set_defaults(niters=10, lda=False)

    args = parser.parse_args()

    classes = load_classes(args.classes)
    #print(classes)

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))

    def f(i):
        data_set = DataSet(0.80, classes)
        for name in eigenvalues:
            ev_list = eigenvalues[name]
            category = name.split('-')[0]
            data_set.add(category, compute_descriptor(ev_list))

        train_set = data_set.get_train_set()
        test_set = data_set.get_test_set()
        if not args.lda:
            classifier = DistanceClassifier(train_set)
        else:
            classifier = LinearClassifier(train_set)
            classifier.train()
        return classifier.evaluation(test_set)

    pool = Pool(4)
    m_test = pool.map(f, [0]*args.niters)

    #print(m_test)
    print("Mean : ")
    print(np.array(m_test, dtype=float).mean())
    print("Standard deviation : ")
    print(np.array(m_test, dtype=float).std())
