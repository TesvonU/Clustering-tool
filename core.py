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


output = read_file("champions.csv")
dataset = output[0]
column = output[1]
line = output[2]
print_dataset(dataset)
print_dataset(column)
print_dataset(line)

output = drop_column(dataset, column[1])
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
save_dataset(dataset, "small_set.csv")
