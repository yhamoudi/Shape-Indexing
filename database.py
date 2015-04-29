import argparse
import pickle
import laplacian
import numpy as np
import random
import scipy.spatial.distance
from multiprocessing import Pool


from sklearn.lda import LDA
from sklearn.externals import joblib


class DataSet:
    def __init__(self, ratio_train):
        self.__ratio_train = ratio_train
        self.__test_set = []
        self.__train_set = []
        self.__classes = {}
        self.__mean = None
        self.__std = None

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
        matrix = np.concatenate(self.__train_set)
        self.__mean = matrix[:, 0:matrix.shape[1]-1].mean(axis=0)
        self.__std = matrix[:, 0:matrix.shape[1]-1].std(axis=0)
        matrix[:, 0:matrix.shape[1]-1] = (matrix[:, 0:matrix.shape[1]-1] - self.__mean) / self.__std
        return matrix

    def get_test_set(self):
        matrix = np.concatenate(self.__test_set)
        matrix[:, 0:matrix.shape[1]-1] = (matrix[:, 0:matrix.shape[1]-1] - self.__mean) / self.__std
        return matrix


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


class LinearClassifier:
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
        estimated_answers = self.__classifier.predict(input_matrix)

        correct_answers = np.sum(estimated_answers == labels)
        ratio_correct_answers = float(correct_answers)/float(labels.shape[0])

        return ratio_correct_answers


class EuclideanClassifier:
    def __init__(self, train_set):
        descriptor_size = train_set.shape[1]-1
        self.__input_matrix = train_set[:, 0:descriptor_size]
        self.__labels = train_set[:, descriptor_size].astype(int)

    def classify(self, normalized_descriptor):
        def f(v):
            return scipy.spatial.distance.euclidean(v, normalized_descriptor)

        result = np.fromiter(map(f, self.__input_matrix),
                             dtype=self.__input_matrix.dtype, count=self.__input_matrix.shape[0])

        return self.__labels[np.argmin(result)]

    def evaluation(self, test_set):
        descriptor_size = test_set.shape[1]-1
        input_matrix = test_set[:, 0:descriptor_size]
        labels = test_set[:, descriptor_size].astype(int)

        estimated_answers = np.fromiter(map(self.classify, input_matrix), dtype=int, count=input_matrix.shape[0])

        correct_answers = np.sum(estimated_answers == labels)
        return float(correct_answers)/float(labels.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the data set')
    parser.add_argument('eigenvalues', metavar='F', help='the file where are stored the eigenvalues')
    args = parser.parse_args()

    eigenvalues = pickle.load(open(args.eigenvalues, "rb"))



    # m_test = []
    # m_train = []
    # for i in range(0, 100):
    #     for name in eigenvalues:
    #         ev_list = eigenvalues[name]
    #         category = name.split('-')[0]
    #         data_set.add(category, laplacian.compute_descriptor(ev_list))
    #
    #     train_set = data_set.get_train_set()
    #     classifier = LinearClassifier(train_set)
    #     classifier.train()
    #     r = data_set.get_test_set()
    #     m_test.append(classifier.evaluation(data_set.get_test_set()))
    #     m_train.append(classifier.evaluation(train_set))
    #
    # print(np.array(m_test, dtype=float).mean())
    # print(np.array(m_train, dtype=float).mean())

    def f(i):
        data_set = DataSet(0.80)
        for name in eigenvalues:
            ev_list = eigenvalues[name]
            category = name.split('-')[0]
            data_set.add(category, laplacian.compute_descriptor(ev_list))

        train_set = data_set.get_train_set()
        test_set = data_set.get_test_set()
        classifier = EuclideanClassifier(train_set)
        return classifier.evaluation(test_set)

    pool = Pool(2)
    m_test = pool.map(f, [0]*10)

    print(m_test)
    print(np.array(m_test, dtype=float).mean())
    print(np.array(m_test, dtype=float).std())