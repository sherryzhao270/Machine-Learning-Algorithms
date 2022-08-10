import numpy as np
import random

class fuzzy_c_means:
    def __init__(self, k: int):
        '''
        kmeans
        '''
        #self._dataset = None
        self._k = k
        self._centroids = None
        self._m = 2

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        self._m = m

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self, ctrd):
        self._centroids = ctrd

    def euclidean_distance(self, vec1, vec2):
        return np.sqrt(np.sum(np.power(vec1 - vec2, 2)))

    def create_centroids(self, dataset):
        n = len(dataset[0])
        dataset = np.array(dataset)
        centroids = np.zeros((self.k, n))
        for j in range(n):
            minJ = np.min(dataset[:,j])
            maxJ = np.max(dataset[:,j])
            #random.seed(10)
            centroids[:,j] = [random.uniform(minJ, maxJ) for i in range(self.k)]
        return centroids

    def find_centroid(self, x, gamma):
        new_centroids = np.zeros(shape = (len(gamma), len(x[0])))
        for i in range(len(gamma)):
            for k in range(len(x[0])):
                new_centroids[i][k] = np.sum(gamma[i]**self.m * x[:,k]) / np.sum(gamma[i]**self.m)
        return new_centroids

    def find_membership_value(self, x, centroids):
        dist = np.zeros(shape = (len(centroids), len(x)))
        for i in range(len(centroids)):
            for k in range(len(x)):
                dist[i][k] = self.euclidean_distance(x[k], centroids[i])
        
        gamma = np.zeros(shape = (len(centroids), len(x)))
        
        for k in range(len(x)):
            for i in range(len(centroids)):
                gamma[i, k] = (np.sum(dist[i,k] ** 2 / np.sum(dist[:,k])) ** (1/(self.m-1))) ** -1
        return gamma

    def clustering(self, dataset):
        precision = 0.01
        maxiter = 100
        iter_num = 0
        step = 1
        centroids = self.create_centroids(dataset)
        while step > precision and iter_num < maxiter:
            gamma = self.find_membership_value(dataset, centroids)
            centroids = self.find_centroid(dataset, gamma)
            step = np.sqrt(np.sum(np.power(gamma, 2)))
            iter_num += 1
        return centroids

    def fit(self, dataset):
        centro = self.clustering(dataset)
        centro = centro[~np.isnan(centro).any(axis=1)]
        #print(centro)
        while len(centro) < 1:
            centro = self.clustering(dataset)
            centro = centro[~np.isnan(centro).any(axis=1)]
        self.centroids = centro
