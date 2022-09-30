

import random


class KMeans:
    def __init__(self, data: list, k=2, max_iteration=300):
        self.k = k
        self.data = data
        self.max_iteration = max_iteration

    def euclidean_distance(self, x1 : list, x2 : list) -> float:
        return sum([(a - b) ** 2 for a, b in zip(x1, x2)]) ** 0.5

    def closest_point(self, x : list, points : list) -> list:
        return min(points, key=lambda p: self.euclidean_distance(x, p))

    def mean(self, points : list) -> list:
        return [sum(x) / len(x) for x in zip(*points)]
    
    def centroids(self) -> list:
        centroids = random.sample(self.data, self.k)
        for _ in range(self.max_iteration):
            clusters = [[] for _ in range(self.k)]
            for x in self.data:
                closest = self.closest_point(x, centroids)
                clusters[centroids.index(closest)].append(x)
            new_centroids = []
            for cluster in clusters:
                new_centroids.append(self.mean(cluster))
            if new_centroids == centroids:
                break
            centroids = new_centroids
        return centroids

  


if __name__ == "__main__":
    k_means = KMeans([[1, 1], [2, 1], [4, 3], [5, 4]]);
    print(k_means.centroids())