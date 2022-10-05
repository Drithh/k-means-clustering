from sklearn.cluster import KMeans as KMeansSklearn
import random


def euclidean_distance(point_1: list, point_2: list) -> float:
    return sum([(a - b) ** 2 for a, b in zip(point_1, point_2)]) ** 0.5


def closest_point(point: list, centroids: list) -> list:
    return min(centroids, key=lambda centroid: euclidean_distance(point, centroid))


def mean(points: list) -> list:
    return [sum(x) / len(x) for x in zip(*points)]


class KMeans:
    def __init__(self, data: list, n_clusters=2, max_iteration=300):
        self.n_clusters = n_clusters
        self.data = data
        self.max_iteration = max_iteration
        self.centroids = self.fit()

    def fit(self) -> list:
        centroids = random.sample(self.data, self.n_clusters)
        for _ in range(self.max_iteration):
            clusters = [[] for _ in range(self.n_clusters)]
            for x in self.data:
                closest = closest_point(x, centroids)
                clusters[centroids.index(closest)].append(x)
            new_centroids = []
            for cluster in clusters:
                new_centroids.append(mean(cluster))
            if new_centroids == centroids:
                self.clusters = clusters
                break
            centroids = new_centroids


def generate_random_data(n=100, k=2, min=0, max=100) -> list:
    data = []
    for _ in range(n):
        data.append([random.randint(min, max) for _ in range(k)])
    return data


if __name__ == "__main__":
    data = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    # print(data)
    kmeans = KMeans(data, n_clusters=2)
    lib_kmeans = KMeansSklearn(n_clusters=2)
    lib_kmeans.fit(data)
    print(lib_kmeans.cluster_centers_)
    print(kmeans.centroids)
    # print(kmeans.clusters)
