import numpy as np
import cv2
from sklearn.decomposition import PCA
# from data_preprocess import *
from pca import *
from sklearn import metrics
import matplotlib.pyplot as plt

import pickle
def write(data, outfile):
    f = open(outfile, "w+b")
    pickle.dump(data, f)
    f.close()

def read(filename):
    f = open(filename)
    data = pickle.load(f)
    f.close()
    return data


#1.read all train data into memory

X_train, Y_train, test_x_orig, test_y,classes= load_data('/')
# train_x_orig_G, train_y_G,test_x_orig, test_y,classes= load_data('/Gaussian_pic/')
# train_x_orig_45, train_y_45,test_x_orig, test_y,classes= load_data('/Rotate/')
# train_x_orig_90, train_y_90,test_x_orig, test_y,classes= load_data('/Rotate_90/')
# train_x_orig_135, train_y_135,test_x_orig, test_y,classes= load_data('/Rotate_135/')

# train_x_orig =X_train #train_x_orig_G+train_x_orig_45+train_x_orig_90+train_x_orig_135

# train_y = np.append(Y_train,train_y_G)
# train_y = np.append(train_y,train_y_45)
# train_y = np.append(train_y,train_y_90)
# train_y = np.append(train_y,train_y_135)

def data_fit_pca(X_train,Y_train):
    #input: original input x,y
    #output: flatten x with y
    train_x=[]
    for item in X_train:
        train_x.append(item.reshape((item.shape[0]*item.shape[1]*3)))
    return np.array(train_x),np.array(Y_train)
train_x, train_y = data_fit_pca(X_train,Y_train)
test_x, test_y =data_fit_pca(test_x_orig, test_y)
#2.applied pca
pca=PCA(0.98)
X_new=pca.fit_transform(train_x)
# print(X_new.shape)
#- See more at: https://shankarmsy.github.io/posts/pca-sklearn.html#sthash.joe4EUjh.dpuf
pca = PCA(n_components=X_new.shape[1])
pca.fit(train_x)
X = pca.transform(train_x)
print("sum_ratio:",np.sum(pca.explained_variance_ratio_)) #sthash.joe4EUjh.dpuf
# print("train:",X)

# print(train_x)

lst=[]
f=open("pca_train.txt",'w')
for x, y in zip(X,train_y):
    f.write(str(x.tolist())[1:-1]+','+str(y)+'\n')
# print(lst)

f.close()

import pandas as pd

#pca for test_Data
pca = PCA(n_components=X_new.shape[1])
pca.fit(test_x)
X_test = pca.transform(test_x)

lst=[]
f=open("pca_test.txt",'w')
for x, y in zip(X_test,test_y):
    f.write(str(x.tolist())[1:-1]+','+str(y)+'\n')
# print(lst)

f.close()

#3.KNN
from sklearn.neighbors import KNeighborsClassifier
test_lst=[]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, train_y)

y_pred = neigh.predict(X)
result=metrics.accuracy_score(y_pred, train_y)

print("------result show-------")
print("train_acc:",result)

# print("Ground truth",train_y)
# print("Train lalel result:",y_pred)

y_pred = neigh.predict(X_test)
result_test=metrics.accuracy_score(y_pred, test_y)
test_lst.append(result_test)
print("test_acc:",result_test)
print("Ground truth",test_y)
print("Test lalel result:",y_pred)

plt.scatter(X_test[:,0], X_test[:,1], c=test_y)
plt.colorbar()
plt.show()

pca_digits = PCA(64).fit(X_test)
plt.semilogx(np.cumsum(pca_digits.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance retained')
plt.ylim(0,1)
plt.show()

'''后续可以处理这个weight达到boosting的效果
weights : str or callable, optional (default = ‘uniform’)
weight function used in prediction. Possible values:
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.'''