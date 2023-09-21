import numpy as np
import pathlib
from tqdm import tqdm
import os
from copy import deepcopy

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import confusion_matrix, accuracy, recall, precision
import torchvision.transforms as T
import torch.nn.functional as F
import argparse

from datasets.basic import BaseDataset
from datasets.unet import UNetBaseDataModule
from models.unet_lightning import UNet
from utils.params import param_parser, dump_params
from metrics.functional import accuracy, precision, recall, iou, dice, true_positives, true_negatives, false_positives, false_negatives
from transforms.BalancedRandomCrop import BalancedRandomCrop


def train_unet(params): 
    # keep copy of all parameters
    all_params = deepcopy(params)
          
    # set up data module
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    num_workers = params["data"].pop("num_workers")
    img_size = params["data"].pop("img_size")

    if "balanced" in params["data"].keys():
        balanced = params["data"].pop("balanced")
    else:
        balanced = False

    if "threshold" in params["data"].keys():
        threshold = params["data"].pop("threshold")
    else:
        threshold = 500
    
    transforms = T.Compose([T.RandomVerticalFlip(),
                            T.RandomHorizontalFlip(),
                            T.RandomRotation(10),
                            #BalancedRandomCrop(img_size, balanced=balanced, threshold=threshold),
                            T.RandomCrop(img_size),
                             ])

    if "norm_mode" in params["data"].keys():
        norm_mode = params["data"].pop("norm_mode")
    else:
        norm_mode = "MinMax"
    
    data_module = UNetBaseDataModule(base_dir=params["data"].pop("base_dir"), 
                                     data_dir=params["data"].pop("data_dir"),
                                     img_size=img_size, # irrelevant,
                                     recursive = params["data"].pop("recursive"),
                                     exclude_tags = params["data"].pop("exclude_tags"),
                                     transforms=transforms,
                                     batch_size=params["experiment"]["batch_size"], 
                                     num_workers=num_workers,
                                     norm_mode=norm_mode,
                                     norm_params={"min": 0, "max": 2**16 -1},
                                     shuffle=params["experiment"]["shuffle"],
                                     seed=params["experiment"]["seed"],
                                     **params["data"])
    
    # create model
    if "in_channels" in params["experiment"].keys():
        in_channels = params["experiment"].pop("in_channels")
    else:
        in_channels = 1
        
    if "out_channels" in params["experiment"].keys():
        out_channels = params["experiment"].pop("out_channels")
    else:
        out_channels = 1

    loss = F.binary_cross_entropy_with_logits
    metrics = [
        accuracy, precision, recall,
        iou, dice, 
        true_positives, true_negatives, 
        false_positives, false_negatives
    ]
    
    if "load_checkpoint" in params["experiment"].keys():
        checkpoint = params["experiment"]["load_checkpoint"]
        model = UNet.load_from_checkpoint(checkpoint_path=checkpoint, 
                                          metrics=metrics,
                                          in_channels=in_channels, 
                                          out_channels=out_channels,
                                          normalization=params["experiment"].pop("layer_norm"), 
                                          lr=params["experiment"].pop("learning_rate"),
                                          **params["experiment"]
                                          )
    else:
        model = UNet(in_channels=in_channels, 
                    out_channels=out_channels, 
                    normalization=params["experiment"].pop("layer_norm"), 
                    lr=params["experiment"].pop("learning_rate"),
                    loss=loss,             #F.nll_loss,
                    metrics=metrics,
                    scheduler=None,
                    **params["experiment"]
                    )
    
    # create logger and model checkpoint callback
    log_path = params["logging"]["path"]
    exp_name = params["experiment"]["name"]
    version = params["experiment"]["version"]
    csv_logger = CSVLogger(save_dir= log_path, 
                           name=exp_name, 
                           version=version)
    
    #checkpoint_dir = params["checkpoints"].pop("path")
    checkpoint_path = os.path.join(params["checkpoints"]["path"], exp_name, "version_{}".format(version), "checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, 
                                          every_n_epochs=params["checkpoints"]["freq"],
                                          save_top_k=-1,
                                          save_last=True)
                
    # copy all parameters in the log directory
    params_path = pathlib.Path(checkpoint_path).parent  / "conf.yml"
    dump_params(params_path, all_params)
                
    # set-up trainer
    resume_checkpoint = None
    if "resume_checkpoint" in params["experiment"].keys():
        resume_checkpoint = params["experiment"]["resume_checkpoint"]

    trainer = Trainer(gpus=AVAIL_GPUS,
                      max_epochs=params["experiment"]["num_epochs"], 
                      progress_bar_refresh_rate=1,
                      logger=[csv_logger], 
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=resume_checkpoint,
                      )
        
    # start the actual training
    trainer.fit(model, datamodule=data_module)
