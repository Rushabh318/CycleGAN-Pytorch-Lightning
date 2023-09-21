import os
import numpy as np
import argparse
import pathlib
from tqdm import tqdm

from training.train_cyclegan import train_cyclegan
from training.train_unet import train_unet

from utils.params import param_parser
from utils.s3_tools import fetch_data, push_data

def debug():
    params = {"data": {"base_dir": "data/cube/annotated_slices/train",
                       "data_dir": {"A": "images", 
                                    "B": "masks"},
                       "recursive": True,
                       "exclude_tags": [],
                       "img_size": 512,
                       "num_workers": 2,
                       "num_classes": 3,
                       "target_type": "class_matrix",
                       },
              "experiment": {"name": "unet_slices_test",
                             "version": 2,
                             "in_channels": 1,
                             "out_channels": 3,
                             "learning_rate": 0.0001,
                             "batch_size": 1,
                             "num_epochs": 10,
                             "layer_norm": "batch",
                             #"norm_output": "LogSoftmax",
                             "shuffle": False,
                             "seed": 42},
              "logging": {"path": "logs/"},
              "checkpoints": {"path": "logs/",
                              "freq": 1}}
    
    return params

if __name__ == "__main__":
    # get set-up parameter
    params = param_parser() 
    #params = debug()
         
    # setup data loader
#     if "num_workers" not in params["data"].keys():
#         params["data"]["num_workers"] = 2
   
    #fetch_data(params["data"]["base_dir"], "../tmp")
    
    #params["experiment"]["load_checkpoint"] = "logs/unet_test_run/version_1/checkpoints/epoch=19-step=79.ckpt"
    
    train_cyclegan(params)
    #train_unet(params)
    
    #push_data("../tmp/logs", "logs")
