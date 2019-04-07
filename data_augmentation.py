'''
Augment pictures
Jemma
@Concordia,Montreal,QC,CA
'''

import cv2
import os
import numpy as np


for path, subdirs, files in os.walk('./lung_data/train/'):
    for name in files:
        file_path = os.path.join(path, name)
        img = cv2.imread(file_path)
        # #Gaussian_filter
        # dst = cv2.GaussianBlur(img, (5, 5), 0)
        # save_path_Gaussian = os.path.join(path,'Gaussian_pic')
        # save_path_Gaussian =  os.path.join(save_path_Gaussian,name)
        # cv2.imwrite(save_path_Gaussian, dst)

        # #Rotate_pictures_90
        rows, cols = img.shape[0],img.shape[1]
        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        # dst = cv2.warpAffine(img, M, (cols, rows))
        # save_path_R90 = os.path.join(path,'Rotate_90')
        # save_path_R90 =  os.path.join(save_path_R90,name)
        # cv2.imwrite(save_path_R90, dst)

        # # Rotate_pictures_45
        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        # dst = cv2.warpAffine(img, M, (cols, rows))
        # save_path_R = os.path.join(path, 'Rotate_45')
        # save_path_R = os.path.join(save_path_R, name)
        # cv2.imwrite(save_path_R, dst)
        # Rotate_pictures_135
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),135, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        save_path_R = os.path.join(path, 'Rotate_135')
        save_path_R = os.path.join(save_path_R, name)
        cv2.imwrite(save_path_R, dst)
        # print(save_path)
