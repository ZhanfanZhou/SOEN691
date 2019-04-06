import random
from itertools import combinations
import numpy as np


# data_list = [[fi, label] or (fi, label)...[fi, label]]
# return rdd: (feature_pattern, [fi, label])
def partition(sc, data_list, total_features, feature_ratio=0.8, sample_ratio=0.8):
    DEFAULT_RKNNS = 15
    # get partition nbr
    selected_features = int(total_features * feature_ratio)
    feature_combos = list(combinations(range(total_features), selected_features))
    max_partitions = len(feature_combos)
    if max_partitions > DEFAULT_RKNNS:
        max_partitions = DEFAULT_RKNNS
    print("combos: ", feature_combos)
    feature_combos = random.sample(feature_combos, k=max_partitions)
    samples_with_tag = []
    for i in range(max_partitions):
        sample = random.sample(data_list, k=int(len(data_list)*sample_ratio))
        # key=feature pattern, value=features
        for s in sample:
            samples_with_tag.append((feature_combos[i], s))
    return sc.parallelize(samples_with_tag)


def distance(vec1, vec2, feature_pattern):
    f_train = []
    f_test = []
    for i in range(vec1.size):
        if i in feature_pattern:
            f_train.append(vec1[i])
            f_test.append(vec2[i])
    return np.sum(np.square(np.array(f_train) - np.array(f_test)))


# [(label, distance)...]
def getKNN(l_d, k):
    labels = [lb[0] for lb in sorted(l_d, key=lambda x: x[1])[:k]]
    return max(labels, key=labels.count)


# rdd: (feature_pattern, [fi, label])
# rdd: (feature_pattern, (label, distance))
# rdd: (feature_pattern, [(label, distance)...])
# rdd: (feature_pattern, classification)
# rdd: classifications
def applyRKNN(tagged, test_sample, k):
    vote = tagged.map(lambda x: (x[0], [(x[1][1], distance(x[1][:-1], test_sample, x[0]))]))\
        .reduceByKey(lambda x, y: x+y)\
        .map(lambda x: getKNN(x[1], k))\
        .map(lambda x: x[1])\
        .collect()
    return max(vote, key=vote.count)


# partition(1, 10, sample_ratio=0.8)
fp = [1,2,3,4,5,6]
tr = np.array([2,234,4,623,53,3,2,9])
te = np.array([21,2,4,4,6,3,2,9])
getKNN([("p", 1), ("n", 11), ("n", 0.4), ("p", 2)], 3)
# sorted(nums)[len(nums)//2]
