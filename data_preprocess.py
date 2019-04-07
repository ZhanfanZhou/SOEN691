#coding=utf-8
#!/usr/bin/python
from __future__ import print_function
import pandas as pd
import numpy as np
import os
from PIL import Image
import h5py

'''
Read csv files and get the label of each picture.
Store the picture data with related label.

Jemma

@Concordia,Montreal,QC,CA
'''

pic_dir = './train'
file_name = './train/Grade.csv'

def check_file(data_dir):
    if os.path.exists(data_dir):
            return True
    else:
        print("No such file,please check the dir!")
        return False


def load_data(subdir):
    """Loads data from CSV file.
    Retrieves a matrix of all rows and columns from Lung  dataset.
    Args:
        file_name
    Returns:
        train_img,train_label,test_img,test_label
    """
    while(check_file(file_name)):
        # FOR training data saving
        data = pd.read_csv(file_name, na_values='_')
        # split test and train data.
        # print(data)
        test_csv = data.values[0:80][:]
        train_csv = data.values[80:165][:]

        train_img = []
        test_img = []
        train_label = []
        test_label = []
        #for path, subdirs, file in os.walk('./lung_data/train/'):
        for file in os.listdir(pic_dir+subdir):
            if (file.startswith('train')):
                img = Image.open(pic_dir+subdir + file)
                img=img.resize((64, 64),Image.BILINEAR)
                #img.show()
                img=np.array(img)
                train_img.append(img)
                for i in range(85):
                    if file.split('.')[0] == train_csv[i][0]:
                        if ( 'malignant' in train_csv[i][2] ):
                            train_label.append(1)
                        elif  ('benign' in train_csv[i][2]) :
                            train_label.append(0)
                        else:
                            train_label.append(2)
        for file in os.listdir(pic_dir+'/test/'):
            if (file.startswith('test')):
                img = Image.open(pic_dir + '/test/' + file)
                img = img.resize((64, 64), Image.BILINEAR)
                img = np.array(img)
                test_img.append(img)
                for i in range(80):
                    if file.split('.')[0] == test_csv[i][0]:
                        #test_label.append(test_csv[i][2])
                        if ( 'malignant' in test_csv[i][2] ):
                            test_label.append(1)
                        elif  ('benign' in test_csv[i][2]) :
                            test_label.append(0)
        train_label=np.array(train_label)
        test_label = np.array(test_label)
        classes = np.array([('malignant'),('benign')])  # the list of classes
        return train_img, train_label, test_img, test_label,classes

# #This part is only for function test.
# if __name__ == "__main__":
#     train_img, train_label, test_img, test_label=load_data()
#     print(type(train_label))
#     print(test_label)
