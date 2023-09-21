import numpy as np
from typing import List, Callable, Tuple
from functools import partial
import random

import torch
import torchvision.transforms.functional as tf

def normalize(inp, mean=None, std=None, eps=1e-8):
    """Normalize based on mean and standard deviation."""
    
    if type(inp) == np.ndarray:
        if mean is None or std is None:
            mean = np.mean(inp)
            std = np.std(inp)
            
        out = (inp - mean) / np.maximum(std, eps)
    elif type(inp) == torch.Tensor:
        if mean is None or std is None:
            mean = torch.mean(inp)
            std = torch.std(inp, unbiased=False)

        out = tf.normalize(inp, [mean], [std])
    else:
        raise TypeError

    return out

def random_crop(img, mask, size, random_key=None):   
    c, h, w = img.shape
    
    if size[0] > h or size[1] > w:
        raise ValueError("Crop is larger than the given image!")

    h_max = h - size[0]
    w_max = w - size[1] 
    
    if random_key is not None:
        random.seed(random_key)
    else:
        random.seed()
        
    top = random.randint(0, h_max)
    left = random.randint(0, w_max)
    
    out_img = tf.crop(img, top, left, height=size[0], width=size[1])
    out_mask = tf.crop(mask, top, left, height=size[0], width=size[1])

    return out_img, out_mask

def random_rotation(img, mask, max_angle, random_key=None):   
    # set seed
    if random_key is not None:
        random.seed(random_key)
    else:
        random.seed()
        
    # rotate in half the cases:
    if random.random() > 0.5:
        return img, mask
    else:        
        angle = random.randint(-max_angle, max_angle)
        
        out_img = tf.affine(img, angle, translate=[0, 0], scale=1, shear=0)
        out_mask = tf.affine(mask, angle, translate=[0, 0], scale=1, shear=0)
   
        return out_img, out_mask

def random_flip(img, mask, direction="horizontal", random_key=None):
     # set seed
    if random_key is not None:
        random.seed(random_key)
    else:
        random.seed()
        
    # flip in half the cases:
    if random.random() < 0.5:
        return img, mask
    else:       
        if direction == "horizontal":
            out_img = tf.hflip(img)
            out_mask = tf.hflip(mask)
        elif direction == "vertical":
            out_img = tf.vflip(img)
            out_mask = tf.vflip(mask)
        else:
            raise KeyError
   
        return out_img, out_mask

def mask_to_pore_ratio(img, mask):
    c, h, w = mask.shape
    pore_ratio = torch.sum(mask) / (h * w) * 100
        
    return img, torch.reshape(pore_ratio, (1,))
    
class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'

class Dual(Repr):
    """Function wrapper to transform two inputs of the same size in the same way"""

    def __init__(self, transform, transform_source: bool = False, transform_target: bool = False, transform_both: bool = False, *args, **kwargs):
        self.transform = partial(transform, *args, **kwargs)
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.transform_both = transform_both

    def __call__(self, source, target, c_idx: int = 0, h_idx: int = 1, w_idx: int = 2):
        #source and target are numpy arrays with shape c, h, w where h and w should be identical
        if self.transform_source:
            return self.transform(source), target
        elif self.transform_target:
            return source, self.transform(target)
        elif self.transform_both or (self.transform_source and self.transform_target):
            assert (source.shape[h_idx:w_idx] == target.shape[h_idx:w_idx]), "Inputs should have the same height and width!"
            source_channels, target_channels = source.shape[c_idx], target.shape[c_idx]
            source_type, target_type = source.type(), target.type()
            both = torch.cat((source, target))
            transformed = self.transform(both)
            source_T, target_T = torch.split(transformed, [source.shape[c_idx], target.shape[c_idx]])
            return source_T.type(source_type), target_T.type(target_type)
        else:
            return source, target

class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)

    def __call__(self, input): 
        return self.function(input)
    
class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, common: bool = False, *args, **kwargs):
        
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target
        self.common = common

    def __call__(self, inp, tar):
        if self.common:
            inp, tar = self.function(inp, tar)
        else:     
            if self.input:
                inp = self.function(inp)
                
            if self.target: 
                tar = self.function(tar)
        
        return inp, tar
    
class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
            
        return inp, target


class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp):
        for t in self.transforms:
            inp = t(inp)
            
        return inp
