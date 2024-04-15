import pandas as pd


def read_file(path):
    dataset = pd.read_csv("path").drop(columns=["Unnamed: 0"])
    return dataset
