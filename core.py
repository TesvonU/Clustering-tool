import pandas as pd
import warnings
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, Birch, AffinityPropagation, SpectralClustering, OPTICS
import hdbscan

# settings for easier console display
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


def read_file(path):
    dataset = pd.read_csv(path)
    columns, lines = read_column_line(dataset)
    print(columns)
    if "Unnamed: 0.1" in columns:
        dataset = pd.read_csv(path).drop(columns=["Unnamed: 0.1"])
    if "Unnamed: 0" in columns:
        dataset = pd.read_csv(path).drop(columns=["Unnamed: 0"])
    print(columns)
    print(dataset)
    return dataset, columns, lines


def save_dataset(dataset, path):
    dataset = dataset.reset_index(drop=True, inplace=False)
    dataset.to_csv(path)


def read_column_line(dataset):
    columns = list(dataset.columns)
    lines = len(dataset)
    return columns, lines


def print_dataset(dataset):
    print(dataset)


def drop_column(dataset, to_drop):
    columns, lines = read_column_line(dataset)
    if to_drop in columns:
        dataset = dataset.drop(columns=[to_drop])
        dataset = dataset.reset_index(drop=True)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def drop_lines(dataset, lower, upper):
    columns, lines = read_column_line(dataset)
    try:
        lower = int(lower)
        upper = int(upper)
    except:
        pass
    if type(lower) is int and type(upper) is int:
        if lower < 0:
            lower = 0
        if lower > lines:
            lower = lines - 1
        if upper < 0:
            upper = 0
        if upper > lines:
            upper = lines - 1
        dataset = dataset.drop(index=range(lower, upper + 1))
        dataset = dataset.reset_index(drop=True)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def sort_dataset(dataset, sort_by):
    dataset = dataset.sort_values(by=sort_by)
    dataset = dataset.reset_index(drop=True)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def drop_duplicates(dataset, unique_value):
    columns, lines = read_column_line(dataset)
    if unique_value not in columns:
        unique_value = "ID"
    dataset = dataset.drop_duplicates(subset=unique_value, keep='first')
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def inpute_nan(dataset, strategy):
    try:
        if strategy == "KNN" or strategy == "knn":
            neigh = 6
            if len(dataset) < 10:
                neigh = 3
            imputer_knn = KNNImputer(n_neighbors=neigh)
            dataset = pd.DataFrame(imputer_knn.fit_transform(dataset), columns=dataset.columns)
            columns, lines = read_column_line(dataset)
            return dataset, columns, lines
        else:
            imputer_simple = SimpleImputer(strategy='mean')
            dataset = pd.DataFrame(imputer_simple.fit_transform(dataset), columns=dataset.columns)
            columns, lines = read_column_line(dataset)
            return dataset, columns, lines
    except:
        return [0, 0, 0, 0]


def remove_anomalies(dataset, percentage):
    columns, lines = read_column_line(dataset)
    columns_to_change = [col for col in columns if col != 'ID']
    dataset_reduced = dataset[columns_to_change].copy()
    percentage_dict = {}
    for column in columns_to_change:
        key = column
        if len(percentage) == 1:
            value = float(percentage[0])
            if value > 0.5:
                value = 0.5
        else:
            value = float(percentage.pop(0))
            if value > 0.5:
                value = 0.5
        if value > 0:
            percentage_dict[key] = value

    for key, value in percentage_dict.items():
        column = dataset_reduced[[key]].copy()
        isolation_forest = IsolationForest(contamination=value, random_state=69)
        isolation_forest.fit(column)
        anomaly_selected = isolation_forest.predict(column) == -1
        dataset_reduced[key][anomaly_selected] = np.nan
    dataset = pd.concat([dataset['ID'], dataset_reduced], axis=1)
    imputer_simple = SimpleImputer(strategy='mean')
    dataset = pd.DataFrame(imputer_simple.fit_transform(dataset), columns=dataset.columns)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def pca_reduction(dataset, dimensions):
    if dimensions == "":
        dimensions = 5
    try:
        dimensions = int(dimensions)
        pca = PCA(n_components=dimensions)
        columns_to_reduce = [col for col in dataset.columns if col != 'ID']
        reduced_data = pca.fit_transform(dataset[columns_to_reduce])
        reduced_dataset = pd.DataFrame(reduced_data, columns=[i for i in range(dimensions)])
        dataset_with_id = pd.concat([dataset['ID'], reduced_dataset], axis=1)
        columns, lines = read_column_line(dataset_with_id)
        return dataset_with_id, columns, lines
    except:
        return [0, 0, 0, 0]


def scale_dataset(dataset):
    try:
        scaler = StandardScaler()
        columns_to_scale = [col for col in dataset.columns if col != 'ID']
        scaled_data = scaler.fit_transform(dataset[columns_to_scale])
        scaled_dataset = pd.DataFrame(scaled_data, columns=columns_to_scale)
        dataset_with_id = pd.concat([dataset['ID'], scaled_dataset], axis=1)
        columns, lines = read_column_line(dataset_with_id)
        return dataset_with_id, columns, lines
    except:
        return [0, 0, 0, 0]


def kmeans_clustering(dataset, n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
    labels = kmeans.fit_predict(dataset)
    return labels


def agglomerative_clustering(dataset, n_clusters=3, affinity='euclidean', linkage='ward'):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = agglomerative.fit_predict(dataset)
    return labels


def dbscan_clustering(dataset, eps=0.5, min_samples=5, metric='euclidean'):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(dataset)
    return labels


def mean_shift_clustering(dataset, bandwidth=0.5, seeds=None, bin_seeding=False, cluster_all=True, min_bin_freq=1, max_iter=300):
    mean_shift = MeanShift(bandwidth=bandwidth, seeds=seeds, bin_seeding=bin_seeding, cluster_all=cluster_all, min_bin_freq=min_bin_freq, max_iter=max_iter)
    labels = mean_shift.fit_predict(dataset)
    return labels


def gaussian_mixture_clustering(dataset, n_components=3, covariance_type='full', tol=1e-3, max_iter=100, n_init=1, init_params='kmeans', random_state=42):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol, max_iter=max_iter, n_init=n_init, init_params=init_params, random_state=random_state)
    labels = gmm.fit_predict(dataset)
    return labels


def birch_clustering(dataset, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True):
    birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters, compute_labels=compute_labels)
    labels = birch.fit_predict(dataset)
    return labels


def affinity_propagation_clustering(dataset, damping=0.5, max_iter=200, convergence_iter=15, preference=None, affinity='euclidean'):
    affinity_propagation = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter, preference=preference, affinity=affinity)
    labels = affinity_propagation.fit_predict(dataset)
    return labels


def spectral_clustering(dataset, n_clusters=3, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans'):
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma, n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels)
    labels = spectral_clustering.fit_predict(dataset)
    return labels


def optics_clustering(dataset, min_samples=5, max_eps=np.inf, metric='minkowski', p=2, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None):
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, p=p, cluster_method=cluster_method, eps=eps, xi=xi, predecessor_correction=predecessor_correction, min_cluster_size=min_cluster_size)
    labels = optics.fit_predict(dataset)
    return labels


def hdbscan_clustering(dataset, min_cluster_size=5, min_samples=5, alpha=1.0, cluster_selection_epsilon=0.0, metric='euclidean', p=None, leaf_size=40, algorithm='best', core_dist_n_jobs=4, allow_single_cluster=False, gen_min_span_tree=False, approx_min_span_tree=True, gen_unexp_graph=False, match_reference_implementation=False, prune_threshold=None, mtree_build_params=None, mtree_leaf_array=None, compact=False, prediction_data=False, cluster_selection_method='eom', cluster_selection_criteria='leaf'):
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, alpha=alpha, cluster_selection_epsilon=cluster_selection_epsilon, metric=metric, p=p, leaf_size=leaf_size, algorithm=algorithm, core_dist_n_jobs=core_dist_n_jobs, allow_single_cluster=allow_single_cluster, gen_min_span_tree=gen_min_span_tree, approx_min_span_tree=approx_min_span_tree, gen_unexp_graph=gen_unexp_graph, match_reference_implementation=match_reference_implementation, prune_threshold=prune_threshold, mtree_build_params=mtree_build_params, mtree_leaf_array=mtree_leaf_array, compact=compact, prediction_data=prediction_data, cluster_selection_method=cluster_selection_method, cluster_selection_criteria=cluster_selection_criteria)
    labels = hdbscan_clusterer.fit_predict(dataset)
    return labels



'''
output = read_file("champions.csv")
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)

output = sort_dataset(dataset, column[1])
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)

output = drop_column(dataset, column[0])
dataset = output[0]
output = drop_column(dataset, column[-1])
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)

output = drop_lines(dataset, 200, 50000)
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)

output = drop_duplicates(dataset, "ID")
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)
save_dataset(dataset, "small_set.csv")
'''
