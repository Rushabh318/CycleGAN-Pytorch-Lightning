import numpy as np
import os
from skimage import io
import pathlib

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

from datasets.basic import DoubleDataset, BaseDataset

class CycleGANDataModule(LightningDataModule):
    
    def __init__(
        self,
        base_dir: str,
        data_dir: dict,
        img_size: int,
        batch_size: int,
        num_workers: int,
        transforms = None,
        norm_mode = None,
        norm_params = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.base_dir = pathlib.Path(base_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.shuffle = shuffle
        self.seed = seed

        # if transforms for both datasets are given seperatly, use them; 
        # otherwise, use the given transforms for both datasets
        if type(transforms) == dict:
            if not all(key in ["A", "B"] for key in transforms.keys()):
                raise KeyError    
        self.transforms = transforms
        
        # check whether data is given
        if data_dir is not None:
            if not type(data_dir) == dict:
                raise TypeError("Dict expected!")
            
            if not all(key in ["A", "B"] for key in data_dir.keys()):
                raise KeyError
            
            self.data_dir = data_dir
        else:
            raise KeyError()
      
        self.norm_mode = norm_mode
        self.norm_params = norm_params

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit': 
            dataset = BaseDataset(datasetA = self.base_dir / self.data_dir["A"],
                                  datasetB = self.base_dir / self.data_dir["B"],
                                  transforms=self.transforms,
                                  norm_mode=self.norm_mode,
                                  norm_params=self.norm_params,
                                  shuffle=self.shuffle,
                                  seed=self.seed)
            
            train_split = int(len(dataset) * 0.8)
            val_split = len(dataset) - train_split
            
            self.dataset_train, self.dataset_val = random_split(dataset, [train_split, val_split],
                                                                torch.Generator().manual_seed(self.seed))
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.dataset_test = BaseDataset(datasetA = self.base_dir / self.data_dir["A"],
                                            datasetB = self.base_dir / self.data_dir["B"],
                                            transforms=self.transforms,
                                            norm_mode=self.norm_mode,
                                            norm_params=self.norm_params,
                                            shuffle=False)
             
            
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)