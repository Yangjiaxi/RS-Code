import numpy as np
from func.util import get_K_neighbors, calc_MAE, calc_RMSE


class UserBasedRS:
    def __init__(self):
        self.train_data_matrix = None
        self.similarity_matrix = None

        self.user_mean_matrix = None

        self.ground_truth = []
        self.prediction = []
        self.rmse = {}
        self.mae = {}

    def clean(self):
        self.ground_truth = []
        self.prediction = []

    def build(self, train_data, builder_func):
        self.train_data_matrix = train_data
        self.similarity_matrix = builder_func(train_data)

        # 避免true_divide除0出错，所以使用ufunc只对分母非零部分处理
        sum_of_pref = train_data.sum(1)
        number_of_rate = (train_data != 0).sum(1)
        self.user_mean_matrix = np.true_divide(sum_of_pref, number_of_rate, out=np.zeros_like(sum_of_pref),
                                               where=(number_of_rate != 0))

    def predict_user_pref(self, item_scores: np.ndarray, user_id, user_pref: np.ndarray, neighbors):
        similarity_sum = np.sum(user_pref[neighbors])
        average_of_user = self.user_mean_matrix[user_id]
        weighted_average = (item_scores[neighbors] - self.user_mean_matrix[neighbors]) @ user_pref[neighbors]

        if similarity_sum == 0:
            return average_of_user
        else:
            return average_of_user + weighted_average / similarity_sum

    def predict(self, test_data: np.ndarray, K, output=False):
        a, b = test_data.nonzero()
        for user_idx, item_idx in zip(a, b):
            neighbors = get_K_neighbors(self.train_data_matrix[:, item_idx], self.similarity_matrix[user_idx], K)
            prediction_once = self.predict_user_pref(self.train_data_matrix[:, item_idx], user_idx,
                                                     self.similarity_matrix[user_idx], neighbors)
            self.ground_truth.append(test_data[user_idx][item_idx])
            self.prediction.append(prediction_once)

        self.rmse[K] = calc_RMSE(self.ground_truth, self.prediction)
        self.mae[K] = calc_MAE(self.ground_truth, self.prediction)

        if output:
            print("User Based: K = %d" % K)
            print("\tRMSE:\t%.6f" % self.rmse[K])
            print("\tMAE :\t%.6f" % self.mae[K])

        return self.rmse[K], self.mae[K]
