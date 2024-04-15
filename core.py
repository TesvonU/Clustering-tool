import pandas as pd
import warnings


# settings for easier console display
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


def read_file(path):
    dataset = pd.read_csv(path)
    columns, lines = read_column_line(dataset)
    return (dataset, columns, lines)


def read_column_line(dataset):
    columns = list(dataset.columns)
    lines = len(dataset)
    return columns, lines


def print_dataset(dataset):
    print(dataset)


def drop_column(dataset, to_drop):
    dataset = dataset.drop(columns=[to_drop])
    columns, lines = read_column_line(dataset)
    return (dataset, columns, lines)


output = read_file("champions.csv")
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)

output = drop_column(dataset, column[-1])
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)
