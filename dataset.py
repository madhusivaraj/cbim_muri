import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CRP(Dataset):
    ''' 
        Only use 150 frames from each clip
    '''
    def __init__(self, args, mode='train', transform=None):
        np.random.seed(args.seed)
        self.model    = args.model
        self.root     = args.root
        self.mode     = mode
        self.temp_stride = args.temp_stride
        if args.debug:
            self.mode = 'debug'

        self.data     = os.listdir(os.path.join(self.root, self.mode))
        self.labels   = {}
        for fname in self.data:
            if 'Spy' in fname:
                self.labels[fname] = 1
            else:
                self.labels[fname] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' 
        Return:
            - features : [T, C] (GRU/LSTM) | [C, T] (TCN)
            - labels   : 0 -> villager, 1 -> spy
        '''
        sample = self.data[idx]
        label  = self.labels[sample]

        data = np.array(np.load(os.path.join(self.root, self.mode, sample))['arr_0'], dtype='float32')
        frames_num = data.shape[0]
        if frames_num <= 150:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, frames_num - 150)
        
        data = data[start_idx:start_idx+150]
        data = data[::self.temp_stride]         # LSTM/GRU: [T, C]
        if self.model=='TCN':
            data = data.T                       # TCN : [C, T]
        
        return data, label
