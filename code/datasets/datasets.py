import numpy as np
import torch
from torch.utils import data
import torchvision.io
import copy

from utils.files import load_img

class PoreAnalysisDataset(data.Dataset):
    def __init__(self, 
                 inputs: list, 
                 targets: list, 
                 transform=None):
        
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index:int):
        img = load_img(self.inputs[index])               
        img_t = np.float32(copy.deepcopy(img))
        
        mask = load_img(self.targets[index])
        mask_t = np.float32(copy.deepcopy(mask))
                  
        # Typecasting
        img_t = torch.from_numpy(img_t).unsqueeze_(0)
        mask_t = torch.from_numpy(mask_t) #.unsqueeze_(0)
                
        if self.transform is not None:
            img_t, mask_t = self.transform(img_t, mask_t.view(1, mask_t.shape[0], mask_t.shape[1]))
            mask_t = torch.squeeze(mask_t, dim=0)
               
        return img_t, mask_t.long()