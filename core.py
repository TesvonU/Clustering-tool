import pandas as pd
import warnings
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# settings for easier console display
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


def read_file(path):
    dataset = pd.read_csv(path)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines

def save_dataset(dataset, path):
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
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def drop_lines(dataset, lower, upper):
    columns, lines = read_column_line(dataset)
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
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def sort_dataset(dataset, sort_by):
    dataset = dataset.sort_values(by=sort_by)
    dataset = dataset.reset_index(drop=True)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def drop_duplicates(dataset, unique_value):
    dataset = dataset.drop_duplicates(subset=unique_value, keep='first')
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def inpute_nan(dataset, strategy):
    if strategy == "KNN":
        neigh = 6
        if len(dataset) < 10:
            neigh = 3
        imputer_knn = KNNImputer(n_neighbors=neigh)
        dataset = pd.DataFrame(imputer_knn.fit_transform(dataset), columns=dataset.columns)
        columns, lines = read_column_line(dataset)
        return dataset, columns, lines
    if strategy == "simple":
        imputer_simple = SimpleImputer(strategy='mean')
        dataset = pd.DataFrame(imputer_simple.fit_transform(dataset), columns=dataset.columns)
        columns, lines = read_column_line(dataset)
        return dataset, columns, lines


def remove_anomalies(dataset, percentage, autoimpute: bool):
    columns, lines = read_column_line(dataset)
    percentage = {
        'HP': 0.01,
        'HP+': 0.01,
        'HP5': 0.01,
        'HP5+': 0.01,
        'MP': 0.01,
        'MP+': 0.01,
        'MP5': 0.001,
        'MP5+': 0.001,
        'AD': 0.001,
        'AD+': 0.001,
        'AS': 0.001,
        'AS+': 0.001,
        'AR': 0.001,
        'AR+': 0.001,
        'MR': 0.001,
        'MR+': 0.001,
        'MS': 0.001,
        'Range': 0.001
    }
    for key, value in percentage.items():
        column = dataset[[key]]
        isolation_forest = IsolationForest(contamination=value,
                                           random_state=69)
        isolation_forest.fit(column)
        # vrací 1/0 jestli je nebo není outlier
        anomaly_selected = isolation_forest.predict(column) == -1
        dataset[key][anomaly_selected] = np.nan

    if autoimpute:
        imputer_simple = SimpleImputer(strategy='mean')
        dataset = pd.DataFrame(imputer_simple.fit_transform(dataset), columns=dataset.columns)

    columns, lines = read_column_line(dataset)
    return dataset, columns, lines


def select_irrelevant(irrelevant: list, to_add: str):
    irrelevant.append(to_add)
    return irrelevant


def scale_dataset(dataset):
    scaler = StandardScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
    columns, lines = read_column_line(dataset)
    return dataset, columns, lines

def pca_reduction(dataset, dimensions: int):
    pca = PCA(n_components=dimensions)
    df_pca = pd.DataFrame(pca.fit_transform(champions_scaled),
                          columns=[i for i in range(dimensions)])

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
