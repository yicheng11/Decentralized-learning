import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ReplayBuffer(object):#we need to store state stansition in a buffer
    '''
    
    This code is copied from openAI baselines
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    '''
    def __init__(self, size):#store experience
        self._storage = []
        self._maxsize = size
        self._next_idx = 0 #put one by one 

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1): #obs_t: current state      obs_tp1: next state
        
        data = (obs_t, action, reward, obs_tp1)

        if self._next_idx >= len(self._storage): # if storage is not full
            self._storage.append(data)
        else: # if storage space is fulled replace the oldest one
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes , dtype = np.float32):
        # store data in list and transfer to np.array
        
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1 = data
            obses_t.append(np.array(obs_t, copy=False,dtype=dtype))
            actions.append(np.array(action, copy=False,dtype=np.long))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False,dtype=dtype))
        return np.array(obses_t,dtype=dtype),np.array(actions , dtype = np.long),np.array(rewards,dtype=dtype),np.array(obses_tp1,dtype=dtype)
    
    
    def sample(self, batch_size): # take randon index from existing data
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class DQN(nn.Module):

    def __init__(self , num_state , num_action):
        
        super().__init__()
        self.fc1 = nn.Linear(num_state , 256 )
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128 , num_action)
        
    def forward(self , x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
