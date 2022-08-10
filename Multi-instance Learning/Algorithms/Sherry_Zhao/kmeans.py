import numpy as np
import random

class kmeans:
    def __init__(self, k: int):
        '''
        kmeans
        '''
        self._dataset = None
        self._k = k
        self._centroids = None

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k

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
            random.seed(20)
            centroids[:,j] = [random.uniform(minJ, maxJ) for i in range(self.k)]
        return centroids

    def find_nearest_cluster(self, dataset, centroids):
        distances = []
        for k in range(self.k):
            distances.append(self.euclidean_distance(dataset, centroids[k]))
        return np.argmin(np.array(distances))

    def clustering(self, dataset):
        n = len(dataset[0])
        previous_centroids = np.zeros((self.k, n))
        centroids = self.create_centroids(dataset)
        threshold = 0.001
        max_iter = 500
        iter_num = 0
        while self.euclidean_distance(previous_centroids, centroids) > threshold and iter_num < max_iter:
            previous_centroids = centroids
            #find the nearest controid for each instance
            clusters = np.zeros((len(dataset)))
            for i in range(len(dataset)):
                clusters[i]= self.find_nearest_cluster(dataset[i], centroids)
            #update centroids
            for k in range(self.k):
                ind = np.where(clusters == k)
                data = np.array(dataset)
                centroids[k, :] = np.mean(data[ind], axis=0)

            iter_num += 1

        return centroids

    def fit(self, dataset):
        centro = self.clustering(dataset)
        centro = centro[~np.isnan(centro).all(axis=1)]
        #print(centro)
        while len(centro) < 1:
            centro = self.clustering(dataset)
            centro = centro[~np.isnan(centro).any(axis=1)]
        self.centroids = centro
