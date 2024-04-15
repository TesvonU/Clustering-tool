import pandas as pd
import warnings


# settings for easier console display
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)


def read_file(path):
    dataset = pd.read_csv(path)
    columns, lines = read_column_line(dataset)
    return (dataset, columns, lines)

def read_column_line(dataset):
    columns = dataset.column
    lines = len(dataset)
    return (columns, lines)

def print_dataset(dataset):
    print(dataset)

output = read_file("champions.csv")
dataset = output[0]
columns = output[1]
lines = output[2]
print_dataset(dataset)
print_dataset(columns)
print_dataset(lines)
