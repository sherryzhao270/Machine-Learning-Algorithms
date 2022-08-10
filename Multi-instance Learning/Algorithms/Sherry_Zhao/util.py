import numpy as np
from mil.data.datasets.loader import load_data

def load_data_no_cv(path):
    return load_data(path)

def normalize(bags):
    b = []
    for bag in bags:
        ins = []
        for instance in bag:
            ins.append([i / np.sum(instance) for i in instance])
        b.append(ins)
    return b

def load_data_cv(path, if_normalize):
    (bags_train, y_train), (bags_test, y_test) = load_data(path)
    bags = []
    bags.extend(bags_train)
    bags.extend(bags_test)
    y = []
    y.extend(y_train)
    y.extend(y_test)
    if if_normalize:
        bags = normalize(bags)
    return bags, y
