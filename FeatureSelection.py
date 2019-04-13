import csv
import random
import numpy as np
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from JobData import get_attrition_data
from sklearn.preprocessing import StandardScaler


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


# data_list = [[array, label] or (array, label)...[array, label]]
# return: (feature_pattern, (fi, label))
def partition(train_data, total_features, feature_ratio=0.9, sample_ratio=0.8, classifiers=5):
    if classifiers > 1:
        selected_features = int(total_features * feature_ratio)
        feature_combos = getCombos(total_features, selected_features, classifiers)
    else:
        feature_combos = [range(total_features)]
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


def RKNN(data_train, data_test, k, dimension, knns):
    spark = init_spark()
    FEATURE_DIM = dimension
    bootstrapping_train = spark.sparkContext.parallelize(partition(data_train, FEATURE_DIM, classifiers=knns))

    res_this_test = []
    for line in open(data_test, "r"):
        line_splitted = line.strip().split(",")
        test_sample = (line_splitted[:-1], line_splitted[-1])
        # test_sample: ([], label)
        res_this_test.append((applyRKNN(bootstrapping_train, test_sample[0], k), test_sample[1]))

    # evaluate = read_in_rdd(spark, data_test)\
    #     .map(lambda x: (applyRKNN(bootstrapping_train, x[0], k), x[1]))\
    #     .map(lambda x: ((x[0], x[1]), 1))\
    #     .reduceByKey(lambda x, y: x+y)\
    #     .collect()

    print("predicted, actual")
    print(res_this_test)
    result_rdd = spark.sparkContext.parallelize(res_this_test)
    evaluate = result_rdd.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda x, y: x+y).collect()
    print(evaluate)


# return: [([], label), ([], label)...([], label)]
def read_data(train_data, test_data, which, dev):
    if which == "cancer":
        train_lst = []
        test_lst = []
        total_features = -1
        for line in open(train_data, "r"):
            sp = line.strip().split(",")
            sp = [float(el) for el in sp]
            total_features = len(sp) - 1
            # ([], label)
            train_lst.append((sp[:-1], sp[-1]))
        for line in open(test_data, "r"):
            sp = line.strip().split(",")
            sp = [float(el) for el in sp]
            # ([], label)
            test_lst.append((sp[:-1], sp[-1]))
        return total_features, train_lst, test_lst, []
    elif which == "heart":
        return read_in_csv(train_data, test_data, dev=dev)
    elif which == "attrition":
        return get_attrition_data(dev=dev)


# data: read in from read_data
# return: all train samples: [[[train_vec1],[train_vec2]...],[]...[]]
def make_data(train_lst, test_lst, dev_lst, total_features, feature_ratio=0.9, sample_ratio=0.8, classifiers=5):
    if classifiers > 1:
        selected_features = int(total_features * feature_ratio)
        feature_combos = getCombos(total_features, selected_features, classifiers)
    else:
        feature_combos = [range(total_features)]

    feature_all_combo = []
    feature_all_combo_test = []
    feature_all_combo_dev = []

    for i in range(classifiers):
        samples = random.sample(train_lst, int(len(train_lst)*sample_ratio))
        feature_this_combo = []
        label_this_combo = []

        feature_this_combo_test = []
        label_this_combo_test = []

        feature_this_combo_dev = []
        label_this_combo_dev = []

        # samples: [([], label), ()...()]
        # s: ([], label)
        for s in samples:
            feature_this_combo.append(make_one_data(s, feature_combos[i]))
            label_this_combo.append(s[1])
        for t in test_lst:
            feature_this_combo_test.append(make_one_data(t, feature_combos[i]))
            label_this_combo_test.append(t[1])

        feature_all_combo.append(feature_this_combo)
        feature_all_combo_test.append(feature_this_combo_test)
        if dev_lst:
            for d in dev_lst:
                feature_this_combo_dev.append(make_one_data(d, feature_combos[i]))
                label_this_combo_dev.append(d[1])
            feature_all_combo_dev.append(feature_this_combo_dev)

    return feature_all_combo, label_this_combo, feature_all_combo_test, label_this_combo_test,\
           feature_combos, feature_all_combo_dev, label_this_combo_dev


def make_one_data(sample, pattern):
    new_feature = []
    for i in range(len(sample[0])):
        if i in pattern:
            new_feature.append(sample[0][i])
    return new_feature


def weighted_vote(vote, boost_y=1, boost_rate=1):
    if boost_rate < 1:
        return vote
    time = 0
    for v in vote:
        if v == boost_y:
            time += 1
    vote += [boost_y]*time*boost_rate
    return vote


def RKNN_sklearn(train_lst, test_lst, feature_nbr, k, feature_ratio=0.8, sample_ratio=0.8, classifiers=5, boost_option=False):
    train_xs, train_y, test_xs, test_y, fc, dev_null, dev_none = make_data(train_lst, test_lst, False, feature_nbr, feature_ratio, sample_ratio, classifiers)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics

    vote = []
    for i in range(classifiers):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_xs[i], train_y)
        y_pred = neigh.predict(test_xs[i])
        vote.append(y_pred.tolist())

    rknn = []
    for i in range(len(vote[0])):
        cur_y = []
        for c in range(classifiers):
            cur_y.append(int(vote[c][i]))
        if boost_option:
            weighted_y = weighted_vote(cur_y, boost_y=1, boost_rate=classifiers-1)
            # print(weighted_y)
        else:
            weighted_y = cur_y
        rknn.append(max(weighted_y, key=weighted_y.count))

    f1 = metrics.f1_score(rknn, test_y)
    # print(rknn, test_y)
    # print("accuracy: %f" % acc)
    print("f1: %f" % f1)
    return f1


# 1.read in data
# 2.run rknn with a set of specific params n times
def rknn_demo(train_path="", test_path="", name="k", rand_time=3, which="cancer", sample_boost=False):
    f_nbr, train, test, dev_null = read_data(train_path, test_path, which=which, dev=False)
    print("default: k=4;fr=0.8;sr=0.8,c=5")
    x = []
    y = []
    if name == "k":
        for k in range(1, 20, 2):
            x.append(k)
            print("this round: k= " + str(k))
            avg_f1 = 0
            for i in range(rand_time):
                avg_f1 += RKNN_sklearn(train, test, f_nbr, k=k, feature_ratio=0.8, sample_ratio=0.8,
                                       classifiers=5, boost_option=sample_boost)
            print("avg: %f" % (avg_f1 / rand_time))
            y.append(avg_f1/rand_time)
        plt.xlabel("k-neighbors(c=5,f_r=0.8,s_r=0.8)")
        plt.xticks(np.arange(min(x), max(x) + 1, 2))
    elif name == "c":
        for k in range(1, 20, 2):
            x.append(k)
            print("this round: classifiers= " + str(k))
            avg_f1 = 0
            for i in range(rand_time):
                avg_f1 += RKNN_sklearn(train, test, f_nbr, k=7, feature_ratio=0.8, sample_ratio=0.8,
                                       classifiers=k, boost_option=sample_boost)
            print("avg: %f" % (avg_f1/rand_time))
            y.append(avg_f1/rand_time)
        plt.xlabel("KNNs(k=7,f_r=0.8,s_r=0.8)")
        plt.xticks(np.arange(min(x), max(x) + 1, 1))
    elif name == "fr":
        for k in range(0, 16):
            r = 0.7 + k/50
            x.append(str(round(r, 2)))
            print("this round: feature_ratio= " + str(r))
            avg_f1 = 0
            for i in range(rand_time):
                avg_f1 += RKNN_sklearn(train, test, f_nbr, k=7, feature_ratio=r, sample_ratio=0.8,
                                       classifiers=5, boost_option=sample_boost)
            print("avg: %f" % (avg_f1 / rand_time))
            y.append(avg_f1/rand_time)

        plt.figure(figsize=(8, 6))
        plt.xlabel("Feature Ratio(k=7,c=5,s_r=0.8)")
    elif name == "sr":
        for k in range(0, 16):
            r = 0.7 + (k*0.02)
            x.append(str(round(r, 2)))
            print("this round: sample_ratio= " + str(r))
            avg_f1 = 0
            for i in range(rand_time):
                avg_f1 += RKNN_sklearn(train, test, f_nbr, k=7, feature_ratio=0.8, sample_ratio=r,
                                       classifiers=5, boost_option=sample_boost)
            print("avg: %f" % (avg_f1 / rand_time))
            y.append(avg_f1/rand_time)

        plt.figure(figsize=(8, 6))
        plt.xlabel("Sample Ratio(k=7,c=5,f_r=0.8)")
    elif name == "all":
        for k in range(5, 6):
            for c in range(1, 10, 2):
                for fr in range(0, 6):
                    for sr in range(0, 6):
                        f_r = 0.80 + (fr * 0.02)
                        s_r = 0.80 + (sr * 0.02)
                        # x.append()
                        print("k=%f,c=%f,fr=%f,sr=%f" % (k, c, f_r, s_r))
                        avg_f1 = 0
                        for i in range(rand_time):
                            avg_f1 += RKNN_sklearn(train, test, f_nbr, k=k, feature_ratio=f_r, sample_ratio=s_r,
                                                   classifiers=c, boost_option=sample_boost)
                        print("avg: %f" % (avg_f1 / rand_time))
                        y.append(avg_f1 / rand_time)
        x = range(len(y))
        plt.figure(figsize=(26, 6))

    plt.bar(np.array(x), np.array(y), width=0.5)
    for a, b in zip(x, y):
        plt.text(a, b + 0.02, "%.3f" % b, ha='center', va='bottom')
    plt.ylim(0, 1.0)
    plt.ylabel("F1-Score")
    plt.title("RKNN_Parameters")
    print("max f1: %f" % max(y))
    print("at: %f" % y.index(max(y)))
    plt.show()


def RKNN_fs(train_lst, test_lst, dev_lst, feature_nbr, k, feature_ratio=0.8, top_ratio=0.8, classifiers=100):
    sample_ratio = 1
    train_xs, train_y, test_xs, test_y, fc, dev_xs, dev_y = make_data(train_lst, test_lst, dev_lst, feature_nbr,
                                                                      feature_ratio, sample_ratio, classifiers)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics

    f_rank = {f: [0, 0] for f in range(feature_nbr)}
    for i in range(classifiers):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_xs[i], train_y)
        y_pred = neigh.predict(test_xs[i])
        f1 = metrics.f1_score(y_pred, test_y)

        for f in fc[i]:
            f_rank[f][0] = (f_rank[f][0] * f_rank[f][1] + f1)/(f_rank[f][1] + 1)
            f_rank[f][1] += 1

    f_info = sorted(f_rank.items(), key=lambda x: x[1][0], reverse=True)
    final_combo = [f[0] for f in f_info][:int(feature_nbr*top_ratio)+1]

    feature_this_combo = []
    label_this_combo = []
    feature_this_combo_test = []
    label_this_combo_test = []

    for s in train_lst:
        feature_this_combo.append(make_one_data(s, final_combo))
        label_this_combo.append(s[1])
    for t in test_lst:
        feature_this_combo_test.append(make_one_data(t, final_combo))
        label_this_combo_test.append(t[1])

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(feature_this_combo, label_this_combo)
    y_pred = neigh.predict(feature_this_combo_test)
    f1 = metrics.f1_score(y_pred, label_this_combo_test)
    return f1


def rknn_fs_demo(train_path="", test_path="", name="c", which="cancer", dev=True):
    f_nbr, train, test, dev = read_data(train_path, test_path, which=which, dev=dev)
    k = 5
    x = []
    y = []
    if name == "fr":
        # 0, 11
        for fr in range(0, 11):
            r = 0.8 + (fr * 0.02)
            x.append(str(round(r, 2)))
            print("this round: feature_ratio= " + str(r))
            avg_f1 = RKNN_fs(train, test, dev, f_nbr, k, feature_ratio=r, top_ratio=0.8, classifiers=50)
            y.append(avg_f1)
            print("f1 = "+str(avg_f1))
        plt.figure(figsize=(8, 6))
        plt.xlabel("Feature Ratio(k=5,c=50,t_r=0.8)")
    elif name == "tr":
        for tr in range(0, 12):
            r = 0.76 + (tr * 0.02)
            x.append(str(round(r, 2)))
            print("this round: top_ratio= " + str(r))
            avg_f1 = RKNN_fs(train, test, dev, f_nbr, k, feature_ratio=0.8, top_ratio=r, classifiers=50)
            y.append(avg_f1)
            print("f1 = " + str(avg_f1))
        plt.figure(figsize=(8, 6))
        plt.xlabel("Top Ratio(k=5,c=50,f_r=0.8)")
    elif name == "c":
        for k in range(30, 90, 10):
            x.append(str(k))
            print("this round: classifiers= " + str(k))
            avg_f1 = RKNN_fs(train, test, dev, f_nbr, k, feature_ratio=0.8, top_ratio=0.8, classifiers=k)
            print("f1 = " + str(avg_f1))
            y.append(avg_f1)
        plt.xlabel("KNNs(k=7,f_r=0.8,s_r=0.8)")

    plt.bar(np.array(x), np.array(y), width=0.5)
    for a, b in zip(x, y):
        plt.text(a, b + 0.02, "%.3f" % b, ha='center', va='bottom')
    plt.ylim(0, 1.0)
    plt.ylabel("F1-Score")
    plt.title("RKNN_Parameters(feature selection)")
    print("max f1: %f" % max(y))
    print("at: %f" % y.index(max(y)))
    plt.show()


def read_in_csv(train_data, test_data, dev):
    if dev:
        dev_file = csv.reader(open("./heart_dev.csv", 'r', encoding="utf-8"))
        train_data = "./heart_train-dev.csv"

    train_file = csv.reader(open(train_data, 'r', encoding="utf-8"))
    test_file = csv.reader(open(test_data, 'r', encoding="utf-8"))
    dev_lst = []
    dev_labels = []
    train_lst = []
    test_lst = []
    train_labels = []
    test_labels = []
    total_features = -1
    for line in train_file:
        sp = [float(el) for el in line]
        total_features = len(sp) - 1
        train_lst.append(sp[:-1])
        train_labels.append(int(sp[-1]))
    for line in test_file:
        sp = [float(el) for el in line]
        test_lst.append(sp[:-1])
        test_labels.append(int(sp[-1]))
    ss = StandardScaler()
    train_lst = ss.fit_transform(train_lst)
    test_lst = ss.fit_transform(test_lst)
    train = []
    test = []
    dev_local = []
    for x, y in zip(train_lst, train_labels):
        train.append((x.tolist(), y))
    for x, y in zip(test_lst, test_labels):
        test.append((x.tolist(), y))

    if dev:
        for line in dev_file:
            sp = [float(el) for el in line]
            dev_lst.append(sp[:-1])
            dev_labels.append(int(sp[-1]))
        dev_lst = ss.fit_transform(dev_lst)
        for x, y in zip(dev_lst, dev_labels):
            dev_local.append((x.tolist(), y))
    return total_features, train, test, dev_local


random.seed(a=66)
# rknn_demo("./pca_train.txt", "./pca_test.txt", "all", rand_time=3, which="cancer")
# rknn_demo("./pca_train_his_pca.txt", "./pca_test_his_pca.txt", "all", rand_time=3, which="cancer")
# rknn_demo("./heart.csv", "./heart2.csv", "all", rand_time=3, which="heart")
rknn_demo(name="c", rand_time=3, which="attrition", sample_boost=True)
# rknn_fs_demo("./pca_train.txt", "./pca_test.txt", name="c", which="cancer")
# rknn_fs_demo("./pca_train_his_pca.txt", "./pca_test_his_pca.txt", name="c", which="cancer")
# rknn_fs_demo("./heart.csv", "./heart2.csv", name="fr", which="heart")
# rknn_fs_demo(name="tr", which="attrition")

# f_nbr, train, test, dev = read_data("", "", which="attrition", dev=False)
# print(RKNN_fs(train, test, dev, f_nbr, 7, feature_ratio=1, top_ratio=1, classifiers=1))
# RKNN_sklearn(train, test, f_nbr, k=7, feature_ratio=1, sample_ratio=1, classifiers=1)
