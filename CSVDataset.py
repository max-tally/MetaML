import csv

import torch
from torch.utils.data import Dataset
import numpy as np

class CSVDataset(Dataset):
    """
    A custom PyTorch Dataset that loads features from one CSV file
    and targets from another CSV file.
    """
    def __init__(self, features_file, targets_file, target_column):
        self.fpath=features_file
        self.tpath=targets_file
        self.features_file = open(self.fpath, 'r')
        self.targets_file = open(self.tpath, 'r')
        self.length = sum(1 for row in self.targets_file) - 1  # Calculate length, subtract header
        self.chunkstart =0
        self.chunksize= 256
        self.targets_file.seek(0)
        self.tchunker= csv.reader(self.targets_file)
        self.fchunker = csv.reader(self.features_file)
        self.theader= next(self.tchunker)
        self.fheader = next(self.fchunker)
        self.target_column = self.theader.index(target_column)
        self.tchunk=self.gettchunk()
        self.fchunk = self.getfchunk()

        print(f'Loaded: {self.features_file} and {self.targets_file} with {self.length} samples. Chunk {self.fchunk.shape} Target column: {self.target_column}')

    def gettchunk(self):
        try:
            arr=np.zeros(self.chunksize,dtype=np.float32)
            for i in range(self.chunksize):
                arr[i]=float(next(self.tchunker)[self.target_column])
            return arr
        except StopIteration:
            return arr[0:i]


    def getfchunk(self):
        try:
            arr=np.ones(shape=(self.chunksize,len(self.fheader)-1),dtype=np.float32)
            for i in range(self.chunksize):
                row=next(self.fchunker)
                arr[i,:]=[self.convert_nan_to_zero(x) for x in row[1:]]
                #arr[i, :] = [self.convert_nan_to_zero(x) for x in row]
            return arr
        except StopIteration:
            return arr[0:i,:]


    def convert_nan_to_zero(self,mystr):
          if mystr == 'NA' or mystr == 'nan':
            return 0
          else:
            return float(mystr)

    def __len__(self):
        return self.length

    def reset(self):
        self.features_file.close()
        self.targets_file.close()
        self.features_file = open(self.fpath, 'r')
        self.targets_file = open(self.tpath, 'r')
        self.fchunker= csv.reader(self.features_file)
        self.tchunker = csv.reader(self.targets_file)
        next(self.tchunker)
        next(self.fchunker)
        self.tchunk=self.gettchunk()
        self.fchunk = self.getfchunk()
        print('Reset files')

    def __getitem__(self, idx):
        if idx < self.length:
            if idx >= self.chunkstart+self.tchunk.shape[0]:
                if not self.targets_file.closed:
                    # load new chunk
                    self.tchunk= self.gettchunk()
                    self.fchunk = self.getfchunk()
                    self.chunkstart += self.tchunk.shape[0]

            try:
                feature = torch.tensor(self.fchunk[idx-self.chunkstart,:])
                target = torch.tensor(self.tchunk[idx-self.chunkstart])
            except IndexError:
                print("End of samples reached. Stopping.")
                raise StopIteration
            if idx % 1000 == 0 :
                print(f'Sample {idx}')
            return feature, target
        else:
            print("End of samples reached. Stopping.")
            raise StopIteration

