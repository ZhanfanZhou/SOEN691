import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from data_preprocess import *
# from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA as RandomizedPCA
img = mpimg.imread('./train/train_1.bmp')
# print(img.shape)
# plt.axis('off')
# plt.imshow(img)

img_r = np.reshape(img, (1, 1213650))
print(img_r.shape)


ipca = RandomizedPCA(1).fit(img_r)
img_c = ipca.transform(img_r)
print(img_c.shape)
print(np.sum(ipca.explained_variance_ratio_))
temp = ipca.inverse_transform(img_c)
print(temp.shape)

plt.axis('off')
plt.imshow(temp)

