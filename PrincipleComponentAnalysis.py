# -*- coding:gbk -*-

from numpy import*
import pandas as pd
import matplotlib.pyplot as plt


class PCA(object):

    def __init__(self):
        pass

    def loadData(self, filename):
        data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=list(range(5)))
        x = data[list(range(4))]
        # print (x)
        return mat(x)

    def pca(self, dataMat, maxFeature=105):
        meanValue = mean(dataMat, axis=0)
        # 去中心，元数据减去均值，值得新的矩阵均值为0
        dataRemMat = dataMat - meanValue
        # 求矩阵的协方差矩阵
        covMat = cov(dataRemMat, rowvar=0)
        # print ("-------covMatt------")
        # print (covMat)
        # 求特征值和特徵向量
        feaValue, feaVect = linalg.eig(mat(covMat))
        # print ("-------特征值-------")
        # print (feaValue)
        # print type(feaValue)
        # print ("-------特征向量-------")
        # print (feaVect)
        # 返回从小到大的索引值print "feaSort" + str(feaValueSort)
        feaValueSort = argsort(feaValue)
        feaValueTopN = feaValueSort[:-(maxFeature + 1):-1]
        redEigVects = feaVect[:, feaValueTopN]  # 选择之后的特征向量矩阵
        # print ("--------TopN特征向量矩阵--------")
        # print (redEigVects)
        # print (shape(redEigVects))
        lowDataMat = dataRemMat * redEigVects  # 数据矩阵*特征向量矩阵 得到降维后的矩阵
        reconMat = lowDataMat * redEigVects.T + meanValue #做数据恢复
        # print (lowDataMat)
        return lowDataMat, reconMat

    def plotW(self, lowDataMat, reconMat):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(lowDataMat[:, 0], lowDataMat[:, 1], marker='*', s=90)
        ax.scatter(reconMat[:, 0], reconMat[:, 1], marker='*', s=50, c='red')
        plt.show()

    def replaceNanWithMean(self):
        datMat = self.loadData('iris.csv')
        numFeat = shape(datMat)[1]
        for i in range(numFeat):
            # values that are not NaN (a number)
            meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
            # set NaN values to mean
            datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal
        print(datMat)
        return datMat


if __name__ == "__main__":
    p = PCA()
    dataMat = p.replaceNanWithMean()
    lowDataMat, reconMat = p.pca(dataMat, 2)
    p.plotW(dataMat, reconMat)