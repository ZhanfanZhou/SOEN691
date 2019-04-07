import random
import numpy as np
from pyspark.sql import SparkSession


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


# data_list = [[array, label] or (array, label)...[array, label]]
# return rdd: (feature_pattern, (fi, label))
def partition(train_data, total_features, feature_ratio=0.8, sample_ratio=0.8, classifiers=5):
    # get partitions
    selected_features = int(total_features * feature_ratio)
    feature_combos = getCombos(total_features, selected_features, classifiers)
    # res_rdd = spark.sparkContext.emptyRDD()
    res = []
    train_lst = []
    for line in open(train_data, "r"):
        sp = line.strip().split(",")
        train_lst.append((sp[:-1], sp[-1]))
    for i in range(classifiers):
        # key=feature, value=label
        # key=feature pattern, value=features,label
        samples = random.sample(train_lst, int(len(train_lst)*sample_ratio))
        for s in samples:
            res.append((feature_combos[i], (s[0], s[1])))
        # sample = train_rdd.takeSample(/True, int(train_rdd.count()*sample_ratio), seed=66)\
        #     .map(lambda x: (feature_combos[i], (x[0], x[1])))
        # res_rdd.union(sample)
    return res


def getCombos(total, pick, partitions):
    random.seed(a=66)
    combos = []
    while len(combos) < partitions:
        s = tuple(random.sample(range(total), pick))
        if s not in combos:
            combos.append(s)
    return combos


def distance(vec1, vec2, feature_pattern):
    f_train = []
    f_test = []
    for i in range(len(vec1)):
        if i in feature_pattern:
            f_train.append(float(vec1[i]))
            f_test.append(float(vec2[i]))
    # return 4.0
    # print(f_train)
    # print(f_test)
    # print(np.array(f_train)-np.array(f_test))

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
    vote = tagged.map(lambda x: (x[0], [(x[1][1], distance(x[1][0], test_sample, x[0]))]))\
        .reduceByKey(lambda x, y: x+y)\
        .map(lambda x: getKNN(x[1], k))\
        .map(lambda x: x[0])\
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
    bootstrapping_train = spark.sparkContext.parallelize(partition(data_train, FEATURE_DIM))
    print(bootstrapping_train.collect())
    res_this_test = []
    for line in open(data_test, "r"):
        line_splitted = line.strip().split(",")
        test_sample = (line_splitted[:-1], line_splitted[-1])
        # print(test_sample[0])
        res_this_test.append((applyRKNN(bootstrapping_train, test_sample[0], k), test_sample[1]))

    # result = read_in_rdd(spark, data_test)\
    #     .map(lambda x: (applyRKNN(bootstrapping_train, x[0], k), x[1]))\
    #     .collect()
    # partition(spark, data_train, 9)
    print("predicted, actual")
    print(res_this_test)


RKNN("./train.txt", "./test.txt", 5, 75)
