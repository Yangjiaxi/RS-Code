import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def divide(title=None):
    print()
    title = "" if title is None else title
    print(title.center(50, "="))
    print()


def format_print_dict(dd, depth=0):
    if not isinstance(dd, dict):
        print(dd)
    else:
        for k, v in dd.items():
            print("\t" * depth, end="")
            print(k, "\t: ", end="")
            if not isinstance(v, dict):
                print(v)
            else:
                print()
                format_print_dict(v, depth + 1)


# 得到前K个最相关的项
def get_K_neighbors(data_vector, similarity_vector, K):
    sim = similarity_vector.copy()
    zero_location = np.where(data_vector == 0)
    sim[zero_location] = 0
    K_neighbors = sparse_matrix_sort(-sim)[0:K]
    return K_neighbors


# 稀疏矩阵排序
def sparse_matrix_sort(matrix):
    non_zero_idx = np.nonzero(matrix)[0]
    res = non_zero_idx[np.argsort(matrix[non_zero_idx])]
    return res


# Rooted Mean Squared Error
def calc_RMSE(ground_truth, pred):
    return np.sqrt(mean_squared_error(ground_truth, pred))


# Mean Absolute Error
def calc_MAE(ground_truth, pred):
    return mean_absolute_error(ground_truth, pred)
