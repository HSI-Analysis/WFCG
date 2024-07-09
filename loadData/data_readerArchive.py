import matplotlib.pyplot as plt
from PIL import Image
import cv2


import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

import sys

class DataReader():
    def __init__(self, dataset_name, traind_on):

        self.data_cube = None
        self.train_gt  = None
        self.test_gt = None


        resized = "Datasets/data_size=696x520/"
        init_path = resized

        path = init_path + dataset_name + "/mat/" + dataset_name 

        raw_data_package = sio.loadmat(path  + ".mat") 
        self.data_cube = raw_data_package[list(raw_data_package.keys())[3]].astype(np.float32)


        shape = self.data_cube.shape
        data_cube_new = np.zeros((shape[1], shape[2], shape[0])) # resised
        for i in range(31):
            data_cube_new[:,:,i]  = self.data_cube[i, :, :]

        self.data_cube = data_cube_new
        train_gt = sio.loadmat(path + "_" + traind_on + ".mat")
        self.train_gt = train_gt[list(train_gt.keys())[3]].astype(np.float32)

        test_gt = sio.loadmat(path + "_gt.mat")
        self.test_gt = test_gt[list(test_gt.keys())[3]].astype(np.float32)

    @property
    def cube(self):
        return self.data_cube
    
    @property
    def test(self):
        return self.test_gt.astype(np.int64)
    
    @property
    def train(self):
        return self.train_gt.astype(np.int64)
    
    @property
    def normal_cube(self):
        return (self.data_cube-np.min(self.data_cube)) / (np.max(self.data_cube)-np.min(self.data_cube))




# PCA
def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca

def data_info(train_label=None, val_label=None, test_label=None, start=1):
    class_num = np.max(train_label.astype('int32'))
    if train_label is not None and val_label is not None and test_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())
        test_mat_num = Counter(test_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i],"\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)
    
    elif train_label is not None and val_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel)
    
    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)
        
    else:
        raise ValueError("labels are None")

def draw(label, name: str = "default", scale: float = 4.0, dpi: int = 400, save_img=None):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)

if __name__ == "__main__":
    data = IndianRaw().cube
    data_gt = IndianRaw().truth
    data_info(data_gt)
    draw(data_gt, save_img=None)
    print(data.shape)
    print(data_gt.shape)





















