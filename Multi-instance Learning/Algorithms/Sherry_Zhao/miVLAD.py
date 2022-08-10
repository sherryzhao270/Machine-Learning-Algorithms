import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import os.path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.linalg import norm

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from kmeans import kmeans 
from fuzzy_c_means import fuzzy_c_means 
from util import *

class miVLADSVM:
    def __init__(self, k, use_soft_assignment):
        '''
        miVLAD
        '''
        self._k = k
        self._codebook = None
        self._model = None
        self._soft_assignment = use_soft_assignment

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k 

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, cdb):
        self._codebook = cdb 

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m

    @property
    def soft_assignment(self):
        return self._soft_assignment

    def learn_codebook(self, learning_bags, k):
        X = []
        for bag in learning_bags:
            for instance in bag:
               X.append(instance)
        if self.soft_assignment:
            X = np.array(X)
            fcm = fuzzy_c_means(k)
            fcm.fit(X)
            return fcm.centroids
        else:
            kms = kmeans(k)
            kms.fit(X)
            self.k = kms.k
            return kms.centroids

    def mapping(self, codebook, bag):
        v_i = np.array([])
        for k in range(len(codebook)):
            v_ik = []
            for l in range(len(bag[0])):
                v_ik.append(np.sum([bag[j][l] - codebook[k][l] for j in range(len(bag))])) 
            v_i = np.concatenate((v_i, v_ik), axis=None)
        return v_i

    def new_feature_vector(self, bags):
        v = []
        for bag in bags:
            v_i = self.mapping(self.codebook, bag)
            v_i = np.sign(v_i) * np.sqrt(np.absolute(v_i))
            v_i = v_i / norm(v_i)
            v.append(v_i)
        return v

    def fit(self, traning_data, train_target):
        self.codebook = self.learn_codebook(traning_data, self.k)
        v = self.new_feature_vector(traning_data)
        self.model = svm.SVC(kernel = "rbf") 
        self.model.fit(v, train_target)

    def predict(self, test_bags):
        v = []
        v = self.new_feature_vector(test_bags)
        prediction = self.model.predict(v)
        return prediction

def evaluate_and_print_metrics(datapath, k, use_soft_assignment):
    bags, y = load_data_cv(datapath, False)
    acc = []
    precision = []
    recall = []
    auc = []
    f1_scores = []
    for i in range(0, 10):
        bags_train, bags_test, y_train, y_test = train_test_split(bags, y, test_size=0.2)
        
        vlad = miVLADSVM(k, use_soft_assignment)
        vlad.fit(bags_train, y_train)
        y_pred = vlad.predict(bags_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))
        f1_scores.append(metrics.f1_score(y_test, y_pred))

    print("miVLAD accuracy", np.mean(acc), "+/-", np.std(acc))
    print("Precision:", np.mean(precision), "+/-", np.std(precision))
    print("Recall:", np.mean(recall), "+/-", np.std(recall))
    print("AUC:", np.mean(auc), "+/-", np.std(auc))
    print("F1-score:", np.mean(f1_scores), "+/-", np.std(f1_scores))
 
 
def miVLAD(datapath, k, use_soft_assignment):
    evaluate_and_print_metrics(datapath, k, use_soft_assignment)

if __name__ == '__main__':
    """
    THIS IS MAIN FUNCTION. 
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a miVLAD algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('-k', help='The number of centroid for codebook.', type=int, const=2, default=2, nargs='?')
    parser.add_argument('--soft-assignment', dest='sa', action='store_true',
                        help='Use soft assignment.')
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    k_clst = args.k
    use_soft_assignment = args.sa
    miVLAD(data_path, k_clst, use_soft_assignment)
    