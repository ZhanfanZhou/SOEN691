import random
from itertools import combinations


# data_list = [[fi, label][fi, label]...[fi, label]]
def bootstrapping_partition(sc, data_list, total_features, feature_ratio=0.8, sample_ratio=0.8):
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
            samples_with_tag.append(feature_combos[i], s)
    return sc.parallelize(samples_with_tag)


def applyRKNN(tagged):
    pass


bootstrapping_partition(1, 10, sample_ratio=0.8)