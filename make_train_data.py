#!/usr/bin/env python
# coding=utf-8
import h5py
from config import FLAGS

import scipy
import scipy.misc
import glob
import os
import numpy as np


class make91dataset(object):
    def __init__(self, raw_data_dir = 'Train'):
        self.raw_data_dir = raw_data_dir
        self.config = FLAGS
        self.save_path = 'train91'
    def prepare_data_files(self):
        filenames = os.listdir(self.raw_data_dir)
        data_dir = os.path.join(os.getcwd(), self.raw_data_dir)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        return data
    def make_h5_data(self):
        patch_sequence = []
        stride = self.config.stride
        input_size = self.config.label_size
        data_filenames = self.prepare_data_files()
        for i in xrange(len(data_filenames)):
            #image = scipy.misc.imread(data_filenames[i], flatten=False, mode = 'RGB').astype(float)
            image = scipy.misc.imread(data_filenames[i], flatten=True, mode = 'YCbCr').astype(float)
            image /=255.0
            h, w = image.shape

            for x in range(0, h-self.config.label_size+1, self.config.stride):
                for y in range(0, w-self.config.label_size+1, self.config.stride):
                    patch_image = image[x:x+self.config.label_size, y:y+self.config.label_size,]
                    patch_image = patch_image.reshape([self.config.label_size, self.config.label_size, 1])
                    patch_sequence.append(patch_image)
        patch_sequence = np.asarray(patch_sequence)

        with h5py.File(self.save_path+'/train_51', 'w') as hf:
            hf.create_dataset('patch_image', data = patch_sequence)


if __name__ =='__main__':
    make_data = make91dataset()
    make_data.make_h5_data()





    
    

