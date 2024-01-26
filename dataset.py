import torch
from torch.utils.data import dataset, dataloader
import csv
import numpy as np
class COVID19Dataset(dataset):
    def __init__(self,
                 path,
                 model = 'train',
                 target_only = False):
        self.model = model

        # Read data from file
        with open(path) as f:
            data = list(csv.reader(f))
            # slice the first row and column
            data = np.array(data[1: ])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            feats = list(range(40)) + [57, 75]

        if model == 'test':
            # Testing data
            # data: 893 * 94
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data : 2700 * 94, train/dev = 9
            target = data[:, -1]
            data = data[:, feats]
            if model == 'train':
                # 行索引，train和dev的比例为9：1
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif model == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features
        self.data[:, 40: ] = (self.data[:, 40: ] - self.data[:, 40:, ].mean(dim=0, keepdim=True)) \
            / self.data[:, 40: ].std(dim=0, keepdim=True)

    def __getitem__(self, index):
        # Return one sample at a time
        if self.model in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no targets)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
