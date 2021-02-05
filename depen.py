#can be deleted, just made for myself to make evrything clean
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random
import torch.nn.functional as F
import os
import pandas as pd
from typing import List, Dict, Tuple
from torch import Tensor
import math, copy, time
from torch.autograd import Variable
from einops import rearrange

import scipy.io
from scipy import signal
import mne
import scipy.linalg


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#seeding can be used when training
def seed_e(seed_value):
  pl.seed_everything(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value) 
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False



def create_channel_list(name, numbers): #"FC", "5/3/1/2/4/6"
  ch_list = list(numbers.replace("/", ""))
  ch_list = [name + s for s in ch_list]
  return ch_list

def find_indicies(ch_list, dict_channels):
  p = []
  for name in ch_list:
    p.append(dict_channels[name])
  return sorted(p)

#b = create_channel_list("FC", "5/3/1/2/4/6")
#find_indicies(b, dict_channels)

def get_mat_file(path, types = 'EEG_MI'):
  mat = scipy.io.loadmat(path)
  mat_train = mat[types + '_train'][0][0]
  mat_test = mat[types + '_test'][0][0]
  return mat_train, mat_test

def get_channel_related_info(mat):
  channels = mat[8][0]
  dict_channels = {}

  for i in range(len(channels)):
    channels[i] = str(channels[i][0])
    dict_channels[channels[i]] = i 
  
  return channels, dict_channels



def get_data(mat, sfrec = 1000, ):
  channels, dict_channels = get_channel_related_info(mat)
  eeg_data = mat[0]
  eeg_data = np.transpose(eeg_data, (1, 2, 0))
  print("eeg shape: ", eeg_data.shape)
  labels = mat[4][0] - 1 # 0 - right, 1 - left # 0 - up, 1 - left, 2 - right, 3 - down
  ch_names = channels.tolist()
  ch_types = ['eeg'] * len(channels)
  info = mne.create_info(ch_names=ch_names, sfreq=sfrec, ch_types=ch_types)
  #
  named_labels = mat[6][0].tolist()
  for i in range(len(named_labels)):
    named_labels[i] = str(named_labels[i][0])
  #
  label_metadata = {'labels_name': named_labels,
          'ids': labels.tolist(),}

  df = pd.DataFrame(label_metadata, columns = ['labels_name', 'ids'])
  epochs_data = mne.EpochsArray(eeg_data, info=info, metadata=df)
  return epochs_data



def get_data_erp(mat, sfrec = 1000, ):
  channels, dict_channels = get_channel_related_info(mat)
  
  a = mat[1].T
  t = mat[2][0]
  
  top = -200
  low = 3800
  eeg_data = np.zeros((t.shape[0], a.shape[0], low - top))
  print(eeg_data.shape)
  i = 0
  for time in t:
    eeg_data[i] = a[:, time+top:time+low]
    i+=1

  print("eeg shape: ", eeg_data.shape)
  labels = mat[4][0] - 1 # 0 - target, 1 - nontarget
  ch_names = channels.tolist()
  ch_types = ['eeg'] * len(channels)
  info = mne.create_info(ch_names=ch_names, sfreq=sfrec, ch_types=ch_types)
  #
  named_labels = mat[6][0].tolist()
  for i in range(len(named_labels)):
    named_labels[i] = str(named_labels[i][0])
  #
  label_metadata = {'labels_name': named_labels,
          'ids': labels.tolist(),}

  df = pd.DataFrame(label_metadata, columns = ['labels_name', 'ids'])
  epochs_data = mne.EpochsArray(eeg_data, info=info, metadata=df)
  return epochs_data