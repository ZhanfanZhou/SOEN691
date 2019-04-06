import random
from itertools import combinations
import numpy as np
from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


# data_list = [[fi, label] or (fi, label)...[fi, label]]
# return rdd: (feature_pattern, (fi, label))
def partition(spark, train_rdd, total_features, feature_ratio=0.8, sample_ratio=0.8):
    DEFAULT_RKNNS = 15
    # get partition nbr
    selected_features = int(total_features * feature_ratio)
    feature_combos = list(combinations(range(total_features), selected_features))
    max_partitions = len(feature_combos)
    if max_partitions > DEFAULT_RKNNS:
        max_partitions = DEFAULT_RKNNS
    feature_combos = random.sample(feature_combos, k=max_partitions)
    res_rdd = spark.sparkContext.emptyRDD()
    for i in range(max_partitions):
        # key=feature, value=label
        # key=feature pattern, value=features,label
        sample = train_rdd.takeSample(True, int(train_rdd.count()*sample_ratio), seed=66)\
            .map(lambda x: (feature_combos[i], (x[0], x[1])))
        res_rdd.union(sample)
    return res_rdd


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


# rdd: (feature_pattern, (fi, label))
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


# rdd: (array, label)...
def read_in_rdd(spark, datafile):
    return spark.read.text(datafile).rdd \
        .map(lambda row: row.value.split(",")) \
        .map(lambda x: (x[:-1], x[-1]))


def RKNN(data_train, data_test, k, dimension):
    spark = init_spark()
    FEATURE_DIM = dimension
    bootstrapping_train = partition(spark, read_in_rdd(spark, data_train), FEATURE_DIM)
    result = read_in_rdd(spark, data_test)\
        .map(lambda x: (applyRKNN(bootstrapping_train, x[0], k), x[1]))\
        .collect()
    print(result)



