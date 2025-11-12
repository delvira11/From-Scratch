import numpy as np
import random
from math import sqrt


def euclidean_distance(point_1, point_2):
    return sqrt(sum((np.array(point_1) - np.array(point_2))**2))

def find_closest_centroid(point, centroids):
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return int(np.argmin(distances))

def points_mean(points):
    n = len(points)
    dims = len(points[0])
    return tuple(sum(point[d] for point in points) / n for d in range(dims))


def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	
    # init_centroids = random.choices(points, k=k)
    # print(init_centroids)

    centroids = initial_centroids # Initialize the centroids (random init, kmeans++ init, etc)

    previous_centroids = map(lambda x: x + 1, centroids)
    while centroids != previous_centroids: # Loop condition until centroids don't move anymore
        clusters = [find_closest_centroid(point, centroids) for point in points] # Assign the closest centroid to every instance to form clusters
        point_cluster = list(zip(points, clusters))
        previous_centroids = centroids
        for j in range (0, k):
            cluster_points = [point[0] for point in point_cluster if point[1] == j]
            centroids[j] = points_mean(cluster_points) # Find the mean all the points that belong to a cluster and make it the new centroid of that cluster
            
    return centroids
        



if __name__ == "__main__":

    # Input
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10

    print(k_means_clustering(points=points, k=k, initial_centroids=initial_centroids, max_iterations=max_iterations))
    print(k_means_clustering([(0, 0, 0), (2, 2, 2), (1, 1, 1), (9, 10, 9), (10, 11, 10), (12, 11, 12)], 2, [(1, 1, 1), (10, 10, 10)], 10))