import os
import numpy as np
import torch
import pathlib

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as T

from datasets.cyclegan import CycleGANDataModule
from models.cyclegan import CycleGAN

def train_cyclegan(params):
    # set up data module
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    num_workers = params["data"]["num_workers"]
    img_size = params["data"]["img_size"]
        
    transforms = T.Compose([T.RandomVerticalFlip(),
                            T.RandomHorizontalFlip(),
                            T.Resize((512, 512)),                          
                            T.RandomCrop(img_size),                            
                             ])
    
    data_module = CycleGANDataModule(base_dir=params["data"]["base_dir"], 
                                     data_dir=params["data"]["data_dir"],
                                     img_size=img_size, # irrelevant
                                     batch_size=params["experiment"]["batch_size"], 
                                     num_workers=num_workers,
                                     norm_mode="MinMax",
                                     norm_params={"min": 0, "max": 2**16 -1},
                                     transforms=transforms,
                                     shuffle=params["experiment"]["shuffle"],
                                     seed=params["experiment"]["seed"])
    data_module.setup(stage='fit')
    
    # create model
    model = CycleGAN(img_sz=img_size, 
                     in_channels=3,
                     epoch_decay=100,
                     **params)
    
    # create logger and model checkpoint callback
    log_path = params["logging"]["path"]
    exp_name = params["experiment"]["name"]
    version = params["experiment"]["version"]
    csv_logger = CSVLogger(save_dir= log_path, 
                           name=exp_name, 
                           version=version)
    
    checkpoint_path = "/".join(["logs", exp_name, "version_{}".format(version), "checkpoints"])
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, 
                                          every_n_epochs=params["checkpoints"]["freq"],
                                          save_top_k=-1,
                                          save_last=True)
    
    # setup trainer
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
    trainer.fit(model, data_module)
