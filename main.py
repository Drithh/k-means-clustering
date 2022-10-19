import csv
import random
# import matplotlib.pyplot as plt
# import sklearn.


def euclidean_distance(point_1: list, point_2: list) -> float:
    return sum([(a - b) ** 2 for a, b in zip(point_1, point_2)]) ** 0.5


def closest_point(point: list, centroids: list) -> list:
    return min(centroids, key=lambda centroid: euclidean_distance(point, centroid))


def mean(points: list) -> list:
    return [sum(x) / len(x) for x in zip(*points)]


class KMeans:
    def __init__(self, data: list, n_clusters=2, max_iteration=300, n_init=10):
        self.n_clusters = n_clusters
        self.data = data
        self.max_iteration = max_iteration
        self.n_init = n_init

        self.centroids = []
        self.clusters = []
        self.inertia = None

    def kmeans(self):
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
                break
            centroids = new_centroids
        return centroids, clusters

    def fit(self):
        print()
        print()

        for i in range(self.n_init):
            centroids, clusters = self.kmeans()
            current_inertia = 0
            for centroid, cluster in zip(centroids, clusters):
                for point in cluster:
                    current_inertia += euclidean_distance(point, centroid) ** 2
            if self.inertia is None or current_inertia < self.inertia:
                self.inertia = current_inertia
                self.centroids = centroids
                self.clusters = clusters


def generate_random_data(n=100, k=2, min=0, max=100) -> list:
    return [[random.randint(min, max) for _ in range(k)] for _ in range(n)]


# def plot_2d(data: list, centroids: list):
#     colors = ['r', 'y', 'b', 'c', 'k', 'g', 'm']
#     for i, centroid in enumerate(centroids):
#         plt.scatter([x[0] for x in data[i]], [x[1]
#                                               for x in data[i]], c=colors[i])
#         plt.scatter(centroid[0], centroid[1], c='black', marker='x')

#     plt.show()


# def plot_3d(clusters: list, centroids: list):
#     colors = ['r', 'y', 'b', 'c', 'k', 'g', 'm']
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for i, cluster in enumerate(clusters):
#         ax.scatter([x[0] for x in cluster], [x[1]
#                                              for x in cluster], [x[2] for x in cluster], c=colors[i])
#         ax.scatter(centroids[i][0], centroids[i][1],
#                    centroids[i][2], c='black', marker='x')
#     plt.show()


if __name__ == "__main__":
    data = []
    with open('data.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append([float(x) for x in row])

    kmeans = KMeans(data, n_clusters=4, n_init=100)
    kmeans.fit()
    for centroid, cluster in zip(kmeans.centroids, kmeans.clusters):
        print(f"Centroid: {centroid}")
        for i, point in enumerate(cluster):
            if i < 1:
                print(f"Cluster:  {point}")
            else:
                print(f"\t  {point}")
        print()

    # without inertia
    for i in range(8):
        kmeans = KMeans(data, n_clusters=4)
        centroids, clusters = kmeans.kmeans()
        print(f"iterasi {i+1}: ", end="")
        for i, centroid in enumerate(centroids):
            # print truncating floating point
            str_data = [f"{x:.2f}" for x in centroid]
            new_data = ""
            min_string_length = 7
            for x in str_data:
                new_data += x + " " * (min_string_length - len(x))
            if i < 1:
                print(f"centroid {i+1}: {new_data}")
            else:
                print(f"{'':11}centroid {i+1}: {new_data}")
        print()

    # with inertia
    for i in range(8):
        kmeans = KMeans(data, n_clusters=4, n_init=100)
        kmeans.fit()
        centroids = sorted(kmeans.centroids)
        print(f"iterasi {i+1}: ", end="")
        for i, centroid in enumerate(centroids):
            # print truncating floating point
            str_data = [f"{x:.2f}" for x in centroid]
            new_data = ""
            min_string_length = 7
            for x in str_data:
                new_data += x + " " * (min_string_length - len(x))
            if i < 1:
                print(f"centroid {i+1}: {new_data}")
            else:
                print(f"{'':11}centroid {i+1}: {new_data}")
        print()
    # print()
    # print()
