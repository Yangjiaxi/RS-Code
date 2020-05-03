from matplotlib import pyplot as plt
from func.util import divide
from itertools import cycle

plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["figure.dpi"] = 250
plt.rcParams["figure.figsize"] = [10, 5]


def plot_exp(run_res, neighbors_list, title="No name"):
    pref = {
        "c": "black",
        "markersize": 4,
        "linewidth": 1.5
    }
    markers = cycle(['o', '*', '^'])
    lines = cycle(['-', '--', ':', 'dashdot'])
    for name, err in run_res.items():
        plt.title(title)
        plt.plot(neighbors_list, err, label=name, **pref, marker=next(markers), linestyle=next(lines))
        plt.legend()

    plt.savefig("output/{}.jpg".format(title))
    plt.show()


def plot_show(run_res, neighbors_list, title="No name"):
    divide("Exp result of `{}`".format(title))

    print(" | {:>14}".format("Method & K"), end="")
    for K in neighbors_list:
        print(" | {:>6}".format(K), end="")
    print(" |")

    for name, errs in run_res.items():
        print(" | {:>14}".format(name), end="")
        for err in errs:
            print(" | {:.4f}".format(err), end="")
        print(" |")
