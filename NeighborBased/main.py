from func.load_data import load_movie_lens, split_data, pandas_to_matrix
from func.similarity import similarity

from model.user_based import UserBasedRS
from model.item_based import ItemBasedRS

from func.plot import plot_exp, plot_show
from func.util import divide

if __name__ == '__main__':

    neighbors = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                 120, 150, 175, 200, 250]

    configs = [
        {
            "kind": "UserBased",
            "similarity": ["cos", "pcc", "jaccard", "msd"],
        },
        {
            "kind": "ItemBased",
            "similarity": ["cos", "msd"],
        },
    ]

    # save result for every configuration
    record = []
    rmse_record = {}
    mae_record = {}

    movie_lens_data = load_movie_lens()
    n_users = movie_lens_data[0].max()
    n_items = movie_lens_data[1].max()

    train_data, test_data = split_data(movie_lens_data, 0.2)
    divide("[ Training set size: {} ]".format(len(train_data)))

    train_matrix = pandas_to_matrix(train_data, n_users, n_items)
    test_matrix = pandas_to_matrix(test_data, n_users, n_items)

    a = (train_matrix != 0).sum()
    b = (test_matrix != 0).sum()

    print("Sparsity: [ %.4f ]" % ((a + b) / (n_users * n_items)))

    for config in configs:
        if config["kind"] == "UserBased":
            model = UserBasedRS()
        elif config["kind"] == "ItemBased":
            model = ItemBasedRS()
        else:
            raise ValueError("Unknown model type: `{}`.\n"
                             "Follow models are valid:\n"
                             "\t`UserBased`\n"
                             "\t`ItemBased`\n".format(config["kind"]))
        for method in config["similarity"]:
            print("Building method `{}` for `{}` model...".format(method, config["kind"]))
            model.build(train_matrix, similarity(method))
            print("\t...build complete")
            name = ("User-" if config["kind"] == "UserBased" else "Item-") + method
            rmse_record[name] = []
            mae_record[name] = []
            for idx, K in enumerate(neighbors):
                print("{:3d}...".format(K), end=("\n" if ((idx + 1) % 5 == 0) else ""), flush=True)
                model.clean()
                rmse, mae = model.predict(test_matrix, K)
                # print("RMSE:\t%.4f\tMSE:\t%.4f" % (rmse, mae))
                rmse_record[name].append(rmse)
                mae_record[name].append(mae)
                record.append(
                    {"kind": config["kind"],
                     "method": method,
                     "K": K,
                     "RMSE": rmse,
                     "MAE": mae}
                )
            divide()
    # print(rmse_record)
    # print(mae_record)

    plot_exp(rmse_record, neighbors, "RMSE-Curve")
    plot_exp(mae_record, neighbors, "MAE-Curve")

    plot_show(rmse_record, neighbors, "RMSE-Report")
    plot_show(mae_record, neighbors, "MAE-Report")
