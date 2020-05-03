import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# 余弦相似度
def cosine(matrix):
    return cosine_similarity(matrix)


# 皮尔逊相关系数
def pcc(matrix):
    return np.corrcoef(matrix)


# 均方差异
def msd(matrix):
    def v_msd(a, b):
        diff = (a - b) ** 2
        bin_a = np.array(a) != 0
        bin_b = np.array(b) != 0
        intersect = bin_a * bin_b
        nums = intersect.sum()
        if nums == 0:
            return 0
        else:
            return (intersect @ diff) / intersect.sum()

    size = len(matrix)
    sim = np.zeros((size, size), dtype=np.float)
    for i in tqdm(range(size)):
        for j in range(size):
            if i != j:
                sim[i, j] = v_msd(matrix[i], matrix[j])
    return 1 - sim / sim.max()

# Jaccard相似度
def jaccard(matrix):
    def v_jaccard(a, b):
        bin_a = np.array(a) != 0
        bin_b = np.array(b) != 0
        intersect = bin_a * bin_b
        union = bin_a + bin_b
        return intersect.sum() / union.sum()

    size = len(matrix)
    sim = np.zeros((size, size), dtype=np.float)
    for i in tqdm(range(size)):
        for j in range(size):
            if i != j:
                sim[i, j] = v_jaccard(matrix[i], matrix[j])
    return sim


def similarity(method):
    if method == "cos":
        return cosine
    elif method == "pcc":
        # Pearson correlation coefficient
        return pcc
    elif method == "jaccard":
        # Jaccard similarity
        return jaccard
    elif method == "msd":
        # Mean squared difference
        return msd
    else:
        raise ValueError("Unknown similarity measure method: `{}`.\n"
                         "Follow methods are valid:\n"
                         "\t`cos`: Cosine similarity\n"
                         "\t`pcc`: Pearson correlation coefficient\n"
                         "\t`jaccard`: Jaccard similarity\n"
                         "\t`msd`: Mean squared difference".format(method))
