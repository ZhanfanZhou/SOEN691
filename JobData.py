import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

path = './attrition_all.csv'


def get_attrition_data(dev):

    sample_data = pd.read_csv(path)

    num_cols = ['Age', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'MonthlyRate', 'DailyRate', 'HourlyRate',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                'Over18', 'OverTime']

    # cat_cols = ["Over18"]

    ord_cols = ['DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                'JobSatisfaction',
                'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

    # 目标列
    target_col = ['Attrition']

    # 所有特征列
    total_cols = num_cols + cat_cols + ord_cols

    used_data = sample_data[total_cols + target_col]

    # 分割训练集，测试集，80%作为训练集，20%作为dev
    # 保证训练集和测试集中的正负样本的比例一样

    # pos_data = used_data[used_data['Attrition'] == 1].reindex()
    # train_pos_data = pos_data.iloc[:int(len(pos_data) * 0.6)].copy()
    # dev_pos_data = pos_data.iloc[int(len(pos_data) * 0.6): int(len(pos_data) * 0.8)].copy()
    # test_pos_data = pos_data.iloc[int(len(pos_data) * 0.8):].copy()
    #
    # neg_data = used_data[used_data['Attrition'] == 0].reindex()
    # train_neg_data = neg_data.iloc[:int(len(neg_data) * 0.6)].copy()
    # dev_neg_data = neg_data.iloc[int(len(pos_data) * 0.6): int(len(pos_data) * 0.8)].copy()
    # test_neg_data = neg_data.iloc[int(len(neg_data) * 0.8):].copy()
    #
    # train_data = pd.concat([train_pos_data, train_neg_data])
    # dev_data = pd.concat([dev_pos_data, dev_neg_data])
    # test_data = pd.concat([test_pos_data, test_neg_data])

    # print('训练集数据个数', len(train_data))
    # print('正负样本比例', len(train_pos_data) / len(train_neg_data))
    # train_data.head()

    # print('测试集数据个数', len(test_data))
    # print('正负样本比例', len(test_pos_data) / len(test_neg_data))
    # test_data.head()

    ss = StandardScaler()
    one_hot_enc = preprocessing.OneHotEncoder()
    # for cat in cat_cols:
    # cat_lab_enc = preprocessing.LabelEncoder()
    # used_cat_feats = cat_lab_enc.fit_transform(used_data[cat_cols]).toarray()

    used_cat_feats = one_hot_enc.fit_transform(used_data[cat_cols]).toarray()
    # print(used_cat_feats)
    # dev_cat_feats = one_hot_enc.fit_transform(used_data[[col+"labeled" for col in cat_cols]]).toarray()
    # test_cat_feats = one_hot_enc.fit_transform(used_data[[col+"labeled" for col in cat_cols]]).toarray()

    # 先进行Label Encoding
    # Gender数据
    # gender_label_enc = preprocessing.LabelEncoder()
    # train_data['Gender_Label'] = gender_label_enc.fit_transform(train_data['Gender'])
    #
    # marital_label_enc = preprocessing.LabelEncoder()
    # train_data['Marital_Label'] = marital_label_enc.fit_transform(train_data['MaritalStatus'])
    #
    # ot_label_enc = preprocessing.LabelEncoder()
    # train_data['OT_Label'] = ot_label_enc.fit_transform(train_data['OverTime'])
    #
    # one_hot_enc = preprocessing.OneHotEncoder()
    # train_cat_feats = one_hot_enc.fit_transform(train_data[['Gender_Label', 'Marital_Label', 'OT_Label']]).toarray()
    # print(train_cat_feats[:5, :])

    # 对测试集数据进行相应的编码操作
    # 注意要使用从训练集中得出的encoder

    # 标签编码
    # Gender数据
    # test_data['Gender_Label'] = gender_label_enc.transform(test_data['Gender'])
    #
    # test_data['Marital_Label'] = marital_label_enc.transform(test_data['MaritalStatus'])
    #
    # test_data['OT_Label'] = ot_label_enc.transform(test_data['OverTime'])
    #
    # test_cat_feats = one_hot_enc.transform(test_data[['Gender_Label', 'Marital_Label', 'OT_Label']]).toarray()

    # 整合所有特征
    used_num_feats = used_data[num_cols].values
    used_ord_feats = used_data[ord_cols].values
    # print(used_num_feats)
    used_feats = np.hstack((used_num_feats, used_ord_feats, used_cat_feats))
    used_feats = ss.fit_transform(used_feats)
    # print(used_feats)
    # used_targets = used_data['Attrition'].values
    used_targets = used_data['Attrition'].map(lambda x: 1 if x == "Yes" else 0).values

    train_lst = []
    test_lst = []
    dev_lst = []
    f_nbr = len(cat_cols + ord_cols + num_cols)+21
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(used_feats, used_targets, test_size=0.25, random_state=66)

    if dev:
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=66)
        for f, l in zip(X_dev, y_dev):
            dev_lst.append((f.tolist(), l))

    # return: [([], label), ([], label)...([], label)]
    for f, l in zip(X_train, y_train):
        train_lst.append((f.tolist(), l))

    for f, l in zip(X_test, y_test):
        test_lst.append((f.tolist(), l))

    # print(train_lst)
    # print(test_lst)
    # X_train = [x.tolist() for x in X_train]
    k = 7
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    test_pred = neigh.predict(X_test)
    print('f1 ：', metrics.f1_score(test_pred, y_test))

    # from FeatureSelection import make_data, make_one_data
    #
    # feature_ratio = 1
    # classifiers = 1
    # top_ratio = 0.8
    # train_xs, train_y, test_xs, test_y, fc, dev_xs, dev_y = make_data(train_lst, test_lst, dev_lst, f_nbr,
    #                                                                   feature_ratio, 1, classifiers)
    #
    # f_rank = {f: [0, 0] for f in range(f_nbr)}
    # for i in range(classifiers):
    #     print(len(train_xs[i][1]))
    #     neigh = KNeighborsClassifier(n_neighbors=k)
    #     neigh.fit(train_xs[i], train_y)
    #     y_pred = neigh.predict(test_xs[i])
    #     f1 = metrics.f1_score(y_pred, test_y)
    #
    #     for f in fc[i]:
    #         f_rank[f][0] = (f_rank[f][0] * f_rank[f][1] + f1)/(f_rank[f][1] + 1)
    #         f_rank[f][1] += 1
    #
    # f_info = sorted(f_rank.items(), key=lambda x: x[1][0], reverse=True)
    # final_combo = [f[0] for f in f_info][:int(f_nbr*top_ratio)+1]
    #
    # feature_this_combo = []
    # label_this_combo = []
    # feature_this_combo_test = []
    # label_this_combo_test = []
    #
    # for s in train_lst:
    #     feature_this_combo.append(make_one_data(s, final_combo))
    #     label_this_combo.append(s[1])
    # for t in test_lst:
    #     feature_this_combo_test.append(make_one_data(t, final_combo))
    #     label_this_combo_test.append(t[1])
    #
    # neigh = KNeighborsClassifier(n_neighbors=k)
    # neigh.fit(feature_this_combo, label_this_combo)
    # y_pred = neigh.predict(feature_this_combo_test)
    # f1 = metrics.f1_score(y_pred, label_this_combo_test)
    # print(f1)

    # for a, b in zip(X_train, feature_this_combo):
    #     if a.tolist() != b:
    #         print(a.tolist())
    #         print(b)

    return f_nbr, train_lst, test_lst, dev_lst

    # # 整合所有特征
    # dev_num_feats = dev_data[num_cols].values
    # dev_ord_feats = dev_data[ord_cols].values
    # dev_feats = np.hstack((dev_num_feats, dev_ord_feats, dev_cat_feats))
    # test_targets = dev_data[target_col].values
    #
    # test_num_feats = test_data[num_cols].values
    # test_ord_feats = test_data[ord_cols].values
    # test_feats = np.hstack((test_num_feats, test_ord_feats, test_cat_feats))
    # test_targets = test_data[target_col].values

    # print('train：', train_feats.shape)
    # print('test：', test_feats.shape)
    # print('dev：', dev_feats.shape)


if __name__ == '__main__':
    get_attrition_data(True)
    # y = [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    # p = [1,0,0,1,0,0,1,0,1,0,0,0,0,0,0]
    # print(metrics.f1_score(y, p))
