import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import os 
import glob
import math

'''
BATCH_SIZE = 8
TRAIN_SPLIT = 0.8

path = '/home/robert/dataset/RWF-2000/RWF-2000_preprocessed'
TRAIN = os.path.join(path, 'train')
TEST = os.path.join(path, 'val')
POSITIVE_DIR = os.path.join(TRAIN, 'Fight')
NEGATIVE_DIR = os.path.join(TRAIN, 'NonFight')
POSITIVE_SAMPLES = glob.glob(os.path.join(POSITIVE_DIR, '*.npy'))
NEGATIVE_SAMPLES = glob.glob(os.path.join(NEGATIVE_DIR, '*.npy'))

POSITIVE_SAMPLES_TRAIN = POSITIVE_SAMPLES[:int(TRAIN_SPLIT*len(POSITIVE_SAMPLES))] 
POSITIVE_SAMPLES_VAL = POSITIVE_SAMPLES[int(TRAIN_SPLIT*len(POSITIVE_SAMPLES)):]

NEGATIVE_SAMPLES_TRAIN = NEGATIVE_SAMPLES[:int(TRAIN_SPLIT*len(NEGATIVE_SAMPLES))]
NEGATIVE_SAMPLES_VAL = NEGATIVE_SAMPLES[int(TRAIN_SPLIT*len(NEGATIVE_SAMPLES)):]
'''



class DataGenerator(nn.Module):
    
    def __init__(self, pos_samples, neg_samples, batch_size, shuffle=False, transforms=None, **kwargs):
        sample = np.load(pos_samples[0])
        sample_shape = sample[:, :, :, :3].shape
        self.pos_samples = list(zip(pos_samples, [1]*len(pos_samples)))
        self.neg_samples = list(zip(neg_samples, [0]*len(neg_samples)))
        self.samples = self.pos_samples + self.neg_samples
        self.transforms = transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ret_shape = (self.batch_size, *sample_shape) # '*' to unpack [batch, (nr_frames, width, height, channels)] => [batch, nr_frames, width...]
        self.epoch_end()

    def epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)
    
    def __len__(self,):
        _len = math.ceil(len(self.samples) / self.batch_size) 
        return _len 

    def __getitem__(self, idx):
        batch_anchor = self.samples[idx*self.batch_size : (idx+1)*self.batch_size]
        
        anchor_samples = np.empty(self.ret_shape, dtype=np.float32)
        anchor_classes = np.empty((self.batch_size, ), dtype=np.uint8)

       	for idx, (file_, class_) in enumerate(batch_anchor):
          arr = np.load(file_)
          arr = arr[:, :, :, :3]
          assert not np.any(np.isnan(arr)),file_
          '''
          if self.transforms:
            for i in range(arr.shape[0]):
               img = to_pil_image(arr[i, :, :, :3])
               img = self.transforms(img)
               img = np.array(img)
               img = img.transpose(1, 2, 0)
               arr[i, :, :, :3] = img
          '''
          anchor_samples[idx, ...] = (arr / 127.5) - 1
          anchor_classes[idx, ...] = class_

          '''
          if self.transforms:
            for i in range(arr.shape[0]):
               img = to_pil_image(arr[i, :, :, :3])
               img = self.transforms(img)
               img = np.array(img)
               img = img.transpose(1, 2, 0)
               arr[i, :, :, :3] = img
          '''
        return anchor_samples, anchor_classes


#data_generator_train = DataGenerator(POSITIVE_SAMPLES_TRAIN, NEGATIVE_SAMPLES_TRAIN, BATCH_SIZE) 
#data_generator_val = DataGenerator(POSITIVE_SAMPLES_VAL, NEGATIVE_SAMPLES_VAL, BATCH_SIZE)
#(anchor_samples, comp_samples), classes = data_generator_train[-1]
#print(f'This is before', anchor_samples.shape)
#print(f'This is it',anchor_samples[0, :, :, :, :])
'''
  The videos contain values in range [0, 255] NEEDS TO BE NORMALIZED [-1, 1] 
for video in data_generator_train.samples:
  f, c = video 
  fil = np.load(f)
  print(fil.shape, fil.dtype, np.min(fil), np.max(fil))
'''
