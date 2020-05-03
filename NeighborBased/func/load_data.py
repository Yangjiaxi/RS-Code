import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_movie_lens():
    # relative to root
    data = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None)
    return data


def split_data(data, ratio=0.25):
    train_data, test_data = train_test_split(data, test_size=ratio)
    return train_data, test_data


def pandas_to_matrix(df, n_users, n_items):
    matrix = np.zeros((n_users, n_items))
    for ele in df.itertuples():
        matrix[ele[1] - 1, ele[2] - 1] = ele[3]
    return matrix
