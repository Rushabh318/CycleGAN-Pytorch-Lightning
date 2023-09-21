import os
import pathlib
import random
from PIL import Image
import numpy as np
import pandas as pd
import zipfile

import matplotlib.pyplot as plt

def load_raw_img(file_path, img_size, file_type, file_offset=None):
    # function to open and convert RAW files 
    
    # open the given file
    with open(file_path, "rb") as f:
        # set up the file
        if file_offset is not None:     
            f.seek(file_offset, os.SEEK_SET)
        
        # convert the image into a numpy array
        img = np.fromfile(f, dtype=file_type)
        img = img.reshape(img_size).astype('float32')
        
    return img

def get_filelist_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as myzip:
        names = myzip.namelist()
        
        return names

def load_img_from_zip(zip_path, img_ind, img_size, file_type):
    with zipfile.ZipFile(zip_path) as myzip:
        names = myzip.namelist()
        
        with myzip.open(names[img_ind], "r") as f:
            content = f.read()
           
            img = np.frombuffer(content, dtype=file_type)
            img = img.reshape(img_size).astype('float32')
            
    return img

def load_img(path, mode=None):
    img = Image.open(path) 
    
    if mode is not None:
        if mode in ["L", "I", "F"]:
            img = img.convert(mode)
        else:
            raise NotImplementedError()
    
    img_np = np.asarray(img)
   
    return img_np

def load_VG_greyvalue_histogram(path):
    content = pd.read_csv(path, sep=",|;", header=None, engine="python")
    
    return np.asarray(content.iloc[:, 2])

def save_img(img, path, mode="I", save_format="PNG"):
        
    if mode == "I":    
        img = np.int32(img)
        out = Image.fromarray(img, mode)
    elif mode == "I;16":    
        img = np.uint16(img)
        out = Image.fromarray(img, mode)
    elif mode == "L":
        img = np.uint8(img)
        out = Image.fromarray(img, mode)
    else:
        raise NotImplementedError("Other formats not implemented yet!")
    
    path = pathlib.Path(path)
    if not path.parent.is_dir():
        os.makedirs(path.parent)
        
    if path.suffix == "":
        path = path.with_suffix("." + save_format)
    
    out.save(path, format=save_format)
    
def train_val_test_split(path, random_key=42, train_share=0.7, val_share=0.1):
    # check whether the given path has two directories "images" and "masks"
    path = pathlib.Path(path)
    if not pathlib.Path.is_dir(path / "images") or not pathlib.Path.is_dir(path / "masks"):
        raise FileNotFoundError("Either images or masks directory is missing (or both)!")
    
    images = sorted((path / "images").glob("*.PNG"))
    masks = sorted((path / "masks").glob("*.PNG"))
    tmp = list(zip(images, masks))
    
    random.seed(random_key)
    random.shuffle(tmp)

    train_split = tmp[0:int(len(tmp)*train_share)]
    val_split = tmp[int(len(tmp)*train_share):int(len(tmp)*(train_share + val_share))]
    test_split = tmp[int(len(tmp)*(train_share + val_share)):]

    out = {"train": {}, "val": {}, "test":{}}
    out["train"]["images"], out["train"]["masks"] = zip(*train_split)
    out["val"]["images"], out["val"]["masks"] = zip(*val_split)
    out["test"]["images"], out["test"]["masks"] = zip(*test_split)
    
    return out

def parse_loss_csv(csv_file):
    data = pd.read_csv(csv_file, index_col="epoch")
    
    train = data[[col for col in data.columns if "train_loss_epoch" in col]]
    val = data[[col for col in data.columns if "val" in col or "epoch" == col]]
    
    num_epochs = np.max(np.unique(data.index)) + 1
  
    output = {"train": {}, "val": {}, "num_epochs": num_epochs}  
    for col in train.columns:   
        if type(train[col][0]) == pd.Series:        
            output["train"][col] = [train[col][i].iloc[-1] for i in range(num_epochs)]
        else: 
            output["train"][col] = train[col]
    
    for col in val.columns:
        if type(val[col][0]) == pd.Series:
            output["val"][col] = [val[col][i].iloc[-2] for i in range(num_epochs)]
        else:
            output["val"][col] = val[col]
    
    return output