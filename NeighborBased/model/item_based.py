import numpy as np
from func.util import get_K_neighbors, calc_MAE, calc_RMSE


class ItemBasedRS:
    def __init__(self):
        self.train_data_matrix = None
        self.similarity_matrix = None

        self.item_mean_matrix = None

        self.ground_truth = []
        self.prediction = []
        self.rmse = {}
        self.mae = {}

    def clean(self):
        self.ground_truth = []
        self.prediction = []

    def build(self, train_data, builder_func):
        self.train_data_matrix = train_data
        self.similarity_matrix = builder_func(train_data.T)

        sum_of_pref = train_data.sum(0)
        number_of_rate = (train_data != 0).sum(0)
        self.item_mean_matrix = np.true_divide(sum_of_pref, number_of_rate, out=np.zeros_like(sum_of_pref),
                                               where=(number_of_rate != 0))

    def predict_user_pref(self, item_scores, item_id, item_pref, K):
        neighbors = get_K_neighbors(item_scores, item_pref, K)

        similarity_sum = np.sum(item_pref[neighbors])
        average_of_item = self.item_mean_matrix[item_id]

        weighted_average = (item_scores[neighbors] - self.item_mean_matrix[neighbors]) @ item_pref[neighbors]
        if similarity_sum == 0:
            return average_of_item
        else:
            return average_of_item + weighted_average / similarity_sum

    def predict(self, test_data, K, output=False):
        a, b = test_data.nonzero()
        for user_idx, item_idx in zip(a, b):
            prediction_once = self.predict_user_pref(self.train_data_matrix[user_idx], item_idx,
                                                     self.similarity_matrix[item_idx], K)
            self.ground_truth.append(test_data[user_idx][item_idx])
            self.prediction.append(prediction_once)

        self.rmse[K] = calc_RMSE(self.ground_truth, self.prediction)
        self.mae[K] = calc_MAE(self.ground_truth, self.prediction)

        if output:
            print("Item Based: K = %d" % K)
            print("\tRMSE:\t%.6f" % self.rmse[K])
            print("\tMAE :\t%.6f" % self.mae[K])

        return self.rmse[K], self.mae[K]
