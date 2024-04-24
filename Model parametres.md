## Clustering Models and Parameters Summary

1. **KMeans Clustering:**
   - `n_clusters`: Number of clusters to form.
     - *Estimated Range:* Typically between 2 and 10.
   - `init`: Method for initialization.
     - Options: `"k-means++"`, `"random"`, or a ndarray.
   - `n_init`: Number of times the k-means algorithm will be run.
     - *Estimated Range:* Typically between 5 and 20.
   - `max_iter`: Maximum number of iterations.
     - *Estimated Range:* Typically between 100 and 500.
   - `tol`: Tolerance to declare convergence.
     - *Estimated Range:* Typically between 1e-4 and 1e-2.
   - `random_state`: Seed for random number generation.
     - *Estimated Range:* Any integer value.

2. **Agglomerative Clustering:**
   - `n_clusters`: The number of clusters to find.
     - *Estimated Range:* Varies based on the dataset and problem.
   - `affinity`: Metric used to compute the linkage.
     - Options: `"euclidean"`, `"l1"`, `"l2"`, `"manhattan"`, `"cosine"`, `"precomputed"`.
   - `linkage`: Linkage criterion to use.
     - Options: `"ward"`, `"complete"`, `"average"`, `"single"`.

3. **DBSCAN Clustering:**
   - `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
     - *Estimated Range:* Typically between 0.1 and 1.0.
   - `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.
     - *Estimated Range:* Typically between 2 and 10.
   - `metric`: The metric to use.
     - Options: `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"cityblock"`, `"cosine"`, `"precomputed"`.

4. **Mean Shift Clustering:**
   - `bandwidth`: Bandwidth parameter.
     - *Estimated Range:* Typically between 0.1 and 1.0.
   - `seeds`: Seeds used to initialize kernels.
   - `bin_seeding`: Whether to seed initial kernels.
   - `cluster_all`: Whether to compute all points.
   - `min_bin_freq`: Minimum number of points in a bin.
   - `max_iter`: Maximum number of iterations.
     - *Estimated Range:* Typically between 100 and 500.

5. **Gaussian Mixture Clustering:**
   - `n_components`: Number of mixture components.
     - *Estimated Range:* Typically between 2 and 10.
   - `covariance_type`: Type of covariance parameters to use.
     - Options: `"full"`, `"tied"`, `"diag"`, `"spherical"`.
   - `tol`: Convergence threshold.
   - `max_iter`: Maximum number of EM iterations.
   - `n_init`: Number of initializations.
   - `init_params`: Method used to initialize parameters.
     - Options: `"kmeans"`, `"random"`.

6. **Birch Clustering:**
   - `threshold`: Branching threshold.
     - *Estimated Range:* Typically between 0.1 and 1.0.
   - `branching_factor`: Maximum number of subclusters to save.
   - `n_clusters`: Number of clusters to find.
   - `compute_labels`: Whether to compute cluster labels.

7. **Affinity Propagation Clustering:**
   - `damping`: Damping factor.
     - *Estimated Range:* Typically between 0.5 and 0.9.
   - `max_iter`: Maximum number of iterations.
   - `convergence_iter`: Number of iterations with no change for convergence.
   - `preference`: Preferences for each point.
     - *Estimated Range:* Often set to the median similarity.
   - `affinity`: Similarity metric used.
     - Options: `"euclidean"`, `"precomputed"`.

8. **Spectral Clustering:**
   - `n_clusters`: Number of clusters to find.
     - *Estimated Range:* Varies based on the dataset and problem.
   - `eigen_solver`: Solver to use.
   - `random_state`: Seed for random number generation.
   - `n_init`: Number of times k-means will be run.

9. **OPTICS Clustering:**
   - `min_samples`: Number of samples in a neighborhood.
     - *Estimated Range:* Typically between 2 and 10.
   - `max_eps`: Maximum distance between two samples.
   - `metric`: Distance metric to use.
     - Options: `"euclidean"`, `"manhattan"`, `"chebyshev"`, `"minkowski"`, `"cityblock"`, `"cosine"`.
   - `cluster_method`: Method used to extract clusters.
     - Options: `"xi"`, `"dbscan"`.
   - `xi`: Determines the minimum steepness on the reachability plot for a cluster boundary.
     - *Estimated Range:* Typically between 0.01 and 0.2.
       

These estimates serve as initial guidelines and may require adjustments based on the dataset characteristics.

