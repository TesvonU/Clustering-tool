## Clustering Models and Parameters Summary

1. **KMeans Clustering:**
   - `n_clusters`: Number of clusters to form.
   - `init`: Method for initialization.
     - Options: `"k-means++"`, `"random"`, or a ndarray.
   - `n_init`: Number of times the k-means algorithm will be run.
   - `max_iter`: Maximum number of iterations.
   - `tol`: Tolerance to declare convergence.
   - `random_state`: Seed for random number generation.

   *Estimated Values:*
   - `n_clusters`: Typically between 2 and 10.

2. **Agglomerative Clustering:**
   - `n_clusters`: The number of clusters to find.
   - `affinity`: Metric used to compute the linkage.
   - `linkage`: Linkage criterion to use.
     - Options: `"ward"`, `"complete"`, `"average"`, `"single"`.

   *Estimated Values:*
   - `n_clusters`: Varies based on the dataset and problem.

3. **DBSCAN Clustering:**
   - `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
   - `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.
   - `metric`: The metric to use.

   *Estimated Values:*
   - `eps`: Typically between 0.1 and 1.0.

4. **Mean Shift Clustering:**
   - `bandwidth`: Bandwidth parameter.
   - `seeds`: Seeds used to initialize kernels.
   - `bin_seeding`: Whether to seed initial kernels.
   - `cluster_all`: Whether to compute all points.
   - `min_bin_freq`: Minimum number of points in a bin.
   - `max_iter`: Maximum number of iterations.

   *Estimated Values:*
   - `bandwidth`: Typically between 0.1 and 1.0.

5. **Gaussian Mixture Clustering:**
   - `n_components`: Number of mixture components.
   - `covariance_type`: Type of covariance parameters to use.
     - Options: `"full"`, `"tied"`, `"diag"`, `"spherical"`.
   - `tol`: Convergence threshold.
   - `max_iter`: Maximum number of EM iterations.
   - `n_init`: Number of initializations.
   - `init_params`: Method used to initialize parameters.
     - Options: `"kmeans"`, `"random"`.

   *Estimated Values:*
   - `n_components`: Typically between 2 and 10.

6. **Birch Clustering:**
   - `threshold`: Branching threshold.
   - `branching_factor`: Maximum number of subclusters to save.
   - `n_clusters`: Number of clusters to find.
   - `compute_labels`: Whether to compute cluster labels.

   *Estimated Values:*
   - `threshold`: Typically between 0.1 and 1.0.

7. **Affinity Propagation Clustering:**
   - `damping`: Damping factor.
   - `max_iter`: Maximum number of iterations.
   - `convergence_iter`: Number of iterations with no change for convergence.
   - `preference`: Preferences for each point.
   - `affinity`: Similarity metric used.

   *Estimated Values:*
   - `damping`: Typically between 0.5 and 0.9.
   - `preference`: Often set to the median similarity.

8. **Spectral Clustering:**
   - `n_clusters`: Number of clusters to find.
   - `eigen_solver`: Solver to use.
   - `random_state`: Seed for random number generation.
   - `n_init`: Number of times k-means will be run.

   *Estimated Values:*
   - `n_clusters`: Varies based on the dataset and problem.

9. **OPTICS Clustering:**
   - `min_samples`: Number of samples in a neighborhood.
   - `max_eps`: Maximum distance between two samples.
   - `metric`: Distance metric to use.
   - `cluster_method`: Method used to extract clusters.
   - `xi`: Determines the minimum steepness on the reachability plot for a cluster boundary.

   *Estimated Values:*
   - `min_samples`: Typically between 2 and 10.
