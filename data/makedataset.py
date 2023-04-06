# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:00:46 2020

@author: Administrator
"""

import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
import argparse
from PIL import Image
class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, file, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset, self).__init__()
		self.trainrgb = trainrgb
		self.trainsyn = trainsyn
		self.train_haze	 = file
		
		h5f = h5py.File(self.train_haze, 'r')
		  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		h5f = h5py.File(self.train_haze, 'r')
		  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)

def data_augmentation(clear ,haze, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    clear = np.transpose(clear, (1, 2, 0))
    haze = np.transpose(haze, (1, 2, 0))
    if mode == 0:
        # original
        clear = clear
        haze = haze
    elif mode == 1:
        # flip up and down
        clear = np.flipud(clear)
        haze = np.flipud(haze)
    elif mode == 2:
        # rotate counterwise 90 degree
        clear = np.rot90(clear)
        haze = np.rot90(haze)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        clear = np.rot90(clear)
        clear = np.flipud(clear)
        haze = np.rot90(haze)
        haze = np.flipud(haze)
    elif mode == 4:
        # rotate 180 degree
        clear = np.rot90(clear, k=2)
        haze = np.rot90(haze, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        clear = np.rot90(clear, k=2)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=2)
        haze = np.flipud(haze)
    elif mode == 6:
        # rotate 270 degree
        clear = np.rot90(clear, k=3)
        haze = np.rot90(haze, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        clear = np.rot90(clear, k=3)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=3)
        haze = np.flipud(haze)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(clear, (2, 0, 1)),np.transpose(haze, (2, 0, 1))

def img_to_patches(img,win,stride,Syn=True):
	chl,raw,col = img.shape
	chl = int(chl)
	num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
	num_col = np.ceil((col-win)/stride+1).astype(np.uint8) 
	count = 0
	total_process = int(num_col)*int(num_raw)
	img_patches = np.zeros([chl,win,win,total_process])
	if Syn:
		for i in range(num_raw):
			for j in range(num_col):			   
				if stride * i + win <= raw and stride * j + win <=col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, stride*j : stride*j + win]				 
				elif stride * i + win > raw and stride * j + win<=col:
					img_patches[:,:,:,count] = img[:,raw-win : raw,stride * j : stride * j + win]		   
				elif stride * i + win <= raw and stride*j + win>col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, col-win : col]
				else:
					img_patches[:,:,:,count] = img[:,raw-win : raw,col-win : col]				
				count +=1		   
	return img_patches

def Train_data(size,stride,path):
    '''synthetic Haze images'''
    train_data = 'NH-Haze20-21-23.h5'
    files2_clear = os.listdir(path+'clear/')
    with h5py.File(train_data, 'w') as h5f:
        count = 0
        scales = [0,0.6,0.7,1.0]
        for i in range(len(files2_clear)):
            hazy_0  = np.array(Image.open(path + 'hazy/' + files2_clear[i])) / 255
            clear_0 = np.array(Image.open(path + 'clear/' + files2_clear[i])) / 255
            print(files2_clear[i])
            for sca in scales:
                print(sca)
                if sca == 0:
                    hazy  = cv2.resize(hazy_0 , (size, size))
                    clear = cv2.resize(clear_0, (size, size))
                else:
                    hazy  = cv2.resize(hazy_0 , (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)
                    clear = cv2.resize(clear_0, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)

                hazy  = img_to_patches(hazy.transpose(2, 0, 1) , size, stride)
                clear = img_to_patches(clear.transpose(2, 0, 1), size, stride)
                for nx in range(clear.shape[3]):
                    clear_out, hazy_out = data_augmentation(clear[:, :, :, nx].copy(), hazy[:, :, :, nx].copy(),
                                                             np.random.randint(0, 7))
                    dataset = np.concatenate([clear_out, hazy_out], 0)
                    h5f.create_dataset(str(count), data=dataset)
                    count += 1
                    print(count, dataset.shape)
    print('Data Num: %d \nData Patch: %d'%(count, size))


        
    h5f.close()	

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description = "Building the training patch database")
    parser.add_argument("--path", "--d", type = str, default='./train/', help="Patch size")
    parser.add_argument("--patch_size", "--p", type = int, default=512, help="Patch size")
    parser.add_argument("--stride", "--s", type = int, default=400, help="Size of stride")
    args = parser.parse_args()
    Train_data(args.patch_size,args.stride,args.path)