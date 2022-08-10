import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import os.path
import warnings
warnings.filterwarnings("ignore")
import timeit

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from util import *

class EMDiverseDensity:

    def __init__(self):
        '''
        EMDD
        '''
        self._best_hypothesis = None
        self._feature_num = None

    @property
    def best_hypothesis(self):
        return self._best_hypothesis

    @best_hypothesis.setter
    def best_hypothesis(self, best_hypothesis):
        self._best_hypothesis = best_hypothesis

    @property
    def feature_num(self):
        return self._feature_num

    @feature_num.setter
    def feature_num(self, feature_num):
        self._feature_num = feature_num

    def distance(self, hypothesis, bag):
        '''
        P(t|B_ij)
        '''
        d = np.sum(np.array(hypothesis[1])**2 * (np.array(bag) - np.array(hypothesis[0]))**2, axis=1)
        return np.exp(-np.array(d))

    def NLDD(self, hypothesis, target_points, target_y):
        nl_dd = 0
        dist = self.distance(hypothesis, target_points)
        
        for i in range(len(target_points)):
            if target_y[i] == 1:
                # if statement deal with runtime issue
                if dist[i] < 1.0e-10:
                    dist[i] = 1.0e-10
                nl_dd += -np.log(dist[i])
            else: 
                if dist[i] < 1.0e-10:
                    dist[i] = 1.0e-10
                nl_dd += -np.log(1 - dist[i])
        return nl_dd
    
    def E_step(self, hypothesis, bags, y):
        p = []
        for bag in bags:
            dist = self.distance(hypothesis, bag)
            p.append(bag[np.argmax(dist)])
        return np.array(p)

    def DD_gradient(self, hypothesis, instances, target):
        hypothesis = np.array(hypothesis)
        grad = np.zeros(shape = (2, len(instances[0])))
        dist = self.distance(hypothesis, instances) 
        for j in range(len(instances)):
            if target[j] == 1:
                grad[0] -= (2 / len(instances[j])) * (hypothesis[1] ** 2) * (instances[j] - hypothesis[0]) 
                grad[1] += (2 / len(instances[j])) * hypothesis[1] * ((instances[j] - hypothesis[0]) ** 2) 
            else:
                grad[0] += (dist[j] / (1 - dist[j])) * (2 / len(instances[j])) * (hypothesis[1] ** 2) * (instances[j] - hypothesis[0]) 
                grad[1] -= (dist[j] / (1 - dist[j])) * (2 / len(instances[j])) * hypothesis[1] * ((instances[j] - hypothesis[0]) ** 2) 
        return grad

    def maximize_DD(self, hypothesis, instance, instance_target):
        '''
        maximize DD with gradient descent
        '''
        learning_rate = 12.35
        threshold = 0.00001
        max_iteration = 5500
        num_iteration = 0
        gradient_norm = 1

        while gradient_norm > threshold and num_iteration < max_iteration:
            grad = self.DD_gradient(hypothesis, instance, instance_target)
            grad = grad / len(instance)
            hypothesis = hypothesis - learning_rate * grad
            gradient_norm = np.sqrt(np.sum(grad[0]**2) + np.sum(grad[1]**2))
            num_iteration += 1
        min_nldd = self.NLDD(hypothesis, instance, instance_target)
        
        return hypothesis, min_nldd

    def fit(self, training_bags, training_target):
        self.feature_num = len(training_bags[0][0])
        pos_ind = np.where(np.array(training_target) == 1)[0]
        bags_selected = pos_ind.argsort()[-5:][::-1]
        pos_bags = [training_bags[bag_selected] for bag_selected in bags_selected]
        min_error = float('inf')

        i = 0

        for pos_bag in pos_bags:
            for instance in pos_bag:
                i += 1
                print('testing hypothesis', i, '...')
                n = len(instance)
                scales = [0.1] * n
                hypothesis = [instance, scales]
                nldd0 = float('inf')
                nldd1 = 999999
                iter_num = 0
                
                while (nldd1 < nldd0 and iter_num < 15):
                    iter_num += 1

                    representative_instance = self.E_step(hypothesis, training_bags, training_target)
                    new_hypothesis, new_nldd = self.maximize_DD(hypothesis, representative_instance, training_target)

                    nldd0 = nldd1
                    nldd1 = new_nldd
                    previous_hypothesis = hypothesis
                    hypothesis = new_hypothesis

                error = 0
                if nldd1 > nldd0:
                    h = previous_hypothesis
                else: 
                    h = hypothesis
                y_pred = self.predict(training_bags, h)
                error = np.count_nonzero(y_pred != training_target)
                if error < min_error:
                    min_error = error
                    self.best_hypothesis = h

    def predict(self, bags, hypothesis = []):
        y_pred = []
        
        for bag in bags:
            if len(hypothesis) <= 0:
                pos = self.distance(self.best_hypothesis, bag)
            else:
                pos = self.distance(hypothesis, bag)
            instance_list = []
            for i in range(len(pos)): 
                if pos[i] >= 0.5:
                    instance_list.append(bag[i])

            if len(instance_list) > 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
                
        return np.array(y_pred)

def selection_by_f_classif(bags, targets):
    X = []
    y = []
    for i in range(len(bags)):
        for instance in bags[i]:
            X.append(instance)
            y.append(targets[i])
    features = SelectKBest(f_classif, k='all').fit(X, y)
    fs = features.scores_
    selected_indices = np.where(fs > 0.7)[0]
    return selected_indices

def feature_selection(bags, selected_indices):
    new_bags = []
    for bag in bags:
        new_bag = []
        for ins in bag:
            new_instance = []
            for selected_ind in selected_indices:
                new_instance.append(ins[selected_ind])
            new_bag.append(new_instance)
        new_bags.append(new_bag)
    return new_bags

def evaluate_and_print_metrics(datapath, select_feature):
    bags, y = load_data_cv(datapath, True)
    acc = []
    precision = []
    recall = []
    auc = []
    time = []
    for i in range(0, 10):
        bags_train, bags_test, y_train, y_test = train_test_split(bags, y, test_size=0.2)
        print("cv", i + 1)
        start = timeit.default_timer()
        if select_feature:
            ind = selection_by_f_classif(bags_train, y_train)
            bags_train = feature_selection(bags_train, ind)
            bags_test = feature_selection(bags_test, ind)
        em_dd = EMDiverseDensity()
        em_dd.fit(bags_train, y_train)
        y_pred = em_dd.predict(bags_test)
        stop = timeit.default_timer()
        acc.append(metrics.accuracy_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))
        time.append(stop - start)

    print("EMDD accuracy", np.mean(acc), "+/-", np.std(acc))
    print("EMDD precision:", np.mean(precision), "+/-", np.std(precision))
    print("EMDD recall:", np.mean(recall), "+/-", np.std(recall))
    print("AUC:", np.mean(auc), "+/-", np.std(auc))
    print("Runtime:", np.mean(time), "+/-", np.std(time))


def EMDD(datapath, select_feature):
    evaluate_and_print_metrics(datapath, select_feature)

if __name__ == '__main__':
    """
    THIS IS MAIN FUNCTION. 
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a miVLAD algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('--feature-selection', dest='fs', action='store_true',
                        help='Use ANOVA F-value to select features.')
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    select_feature = args.fs
    EMDD(data_path, select_feature)