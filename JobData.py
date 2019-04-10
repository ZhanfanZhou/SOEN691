import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

path = './attrition_all.csv'


def get_attrition_data(dev):

    sample_data = pd.read_csv(path)

    num_cols = ['Age', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'MonthlyRate', 'DailyRate', 'HourlyRate',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
                'Over18', 'OverTime']

    # cat_cols = ['Gender', 'MaritalStatus', 'OverTime']
    cat_cols = []

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
    # for cat in cat_cols:
    # cat_lab_enc = preprocessing.LabelEncoder()
    # used_cat_feats = cat_lab_enc.fit_transform(used_data[cat_cols]).toarray()

    # one_hot_enc = preprocessing.OneHotEncoder()
    # used_cat_feats = one_hot_enc.fit_transform(used_data[cat_cols]).toarray()
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
    # used_cat_feats = used_data[cat_cols].values
    used_feats = np.hstack((used_num_feats, used_ord_feats))
    used_feats = ss.fit_transform(used_feats)
    # print(used_feats)
    used_targets = used_data[target_col].values

    train_lst = []
    test_lst = []
    dev_lst = []
    f_nbr = len(cat_cols + ord_cols + num_cols)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(used_feats, used_targets, test_size=0.25, random_state=66)

    if dev:
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=66)
        for f, l in zip(X_dev, y_dev):
            if l[0] == "No":
                dev_lst.append((f, 0))
            else:
                dev_lst.append((f, 1))
        print(dev_lst)

    # return: [([], label), ([], label)...([], label)]
    for f, l in zip(X_train, y_train):
        if l[0] == "No":
            train_lst.append((f, 0))
        else:
            train_lst.append((f, 1))
    for f, l in zip(X_test, y_test):
        if l[0] == "No":
            test_lst.append((f, 0))
        else:
            test_lst.append((f, 1))
    # print(train_lst)
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
