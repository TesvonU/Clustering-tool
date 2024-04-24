KMeans Clustering:
n_clusters: Number of clusters to form.
init: Method for initialization.
Options: 'k-means++' (default), 'random'.
n_init: Number of times the k-means algorithm will be run with different centroid seeds.
max_iter: Maximum number of iterations of the k-means algorithm for a single run.
tol: Relative tolerance with regards to inertia to declare convergence.
random_state: Seed for the random number generator.
Estimated Values:
n_clusters: Typically between 2 and 10.
n_init: Typically between 5 and 20.
max_iter: Generally around 100 to 300.
Agglomerative Clustering:
n_clusters: The number of clusters to find.
affinity: Metric used to compute the linkage.
linkage: Linkage criterion to determine the merging strategy.
Estimated Values:
n_clusters: Varies based on the dataset and problem.
DBSCAN Clustering:
eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
metric: The metric to use when calculating distance between instances in a feature array.
Estimated Values:
eps: Depends on the dataset and desired cluster size.
min_samples: Typically between 2 and 10.
Mean Shift Clustering:
bandwidth: The bandwidth used in the RBF kernel.
seeds: Seeds used to initialize kernel locations.
bin_seeding: Whether to seed using bin means of data.
cluster_all: Whether to compute all points in the dataset.
min_bin_freq: Minimum number of points in a bin to consider it a peak.
Estimated Values:
bandwidth: Typically determined using a cross-validation approach.
Gaussian Mixture Clustering:
n_components: The number of mixture components.
covariance_type: The type of covariance parameters to use.
Options: 'full', 'tied', 'diag', 'spherical'.
tol: The convergence threshold.
max_iter: The maximum number of EM iterations.
n_init: The number of initializations to perform.
init_params: The method used to initialize the weights, means, and precisions.
Options: 'kmeans', 'random'.
Estimated Values:
n_components: Typically between 2 and 10.
Birch Clustering:
threshold: The branching threshold.
branching_factor: The maximum number of subclusters to save.
n_clusters: The number of clusters to find.
compute_labels: Whether to compute cluster labels.
Estimated Values:
threshold: Typically between 0.1 and 1.0.
Affinity Propagation Clustering:
damping: Damping factor.
max_iter: Maximum number of iterations.
convergence_iter: Number of iterations with no change to declare convergence.
preference: Preferences for each point.
affinity: Similarity metric used.
Estimated Values:
damping: Typically between 0.5 and 0.9.
preference: Often set to the median similarity.
Spectral Clustering:
n_clusters: The number of clusters to find.
eigen_solver: Solver to use.
random_state: Seed for random number generation.
n_init: Number of times k-means will be run.
Estimated Values:
n_clusters: Varies based on the dataset and problem.
OPTICS Clustering:
min_samples: The number of samples in a neighborhood.
max_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
metric: The distance metric to use.
cluster_method: The method used to extract clusters from the computed reachability.
xi: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
Estimated Values:
min_samples: Typically between 2 and 10.
