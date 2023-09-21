import os
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torchvision

import torchvision.transforms as T
from datasets.cyclegan import CycleGANDataModule
from models.cyclegan import CycleGAN
from utils.files import save_img
from utils.plots import plot_img_comp, plot_img

import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

def gen_mask(size, seed: int=None):
    mask = np.zeros((size, size))
    
    if seed is not None:
        np.random.seed(seed)
    
    # get random number of pores
    number_pores = np.random.randint(20, 50)
    
    for i in range(number_pores):
        # get random position
        x,y = np.random.randint(0, size-7, 2)
      
        # get random radius        
        r = np.random.randint(1, 5)
      
        for k in range(-r, r+1):
            for j in range(-r, r+1):
                if np.sqrt(k**2 + j**2) < r:
                    mask[x + k, y + j] = 1
        
    return np.float32(mask)

def gen_fake_images(args):
    # set basic parameter
    img_size = args["img_size"]
    batch_size = args["batch_size"]
    
    # load model
    checkpoint_path =  args["checkpoint_path"]    
    model = CycleGAN.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                          img_sz=img_size,
                                          in_channels=1)
    
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        model.cuda()
    
    image_counter = 0
    for i in tqdm(range(args["num_img"])):    
        
        with torch.no_grad():
            # create a simple fake mask
            fake_mask = torch.zeros(batch_size, 1, img_size, img_size)
            if cuda_avail:
                fake_mask = fake_mask.cuda()
            
            for j in range(batch_size):
                fake_mask[j] = torch.as_tensor(gen_mask(img_size)).unsqueeze(dim=0)
                    
            # generate a fake image based on the fake mask
            _, fake_B = model.forward(torch.zeros_like(fake_mask), fake_mask)
            
            # generate a binary mask based for the generated fake image
            fake_A, _ = model.forward(fake_B, torch.zeros_like(fake_mask))
            fake_A_thr = torch.where(torch.abs(fake_A) > 0.9, 1, 0).to(torch.float32)
           
            # generate a fake image once againg using the generated (more realistic) fake mask
            _, fake_B2 = model.forward(torch.zeros_like(fake_mask), fake_A_thr)
            
        # convert to numpy for saving/plotting 
        for j in range(batch_size):
            mask = fake_mask[j][0].cpu().numpy()
            img = fake_B[j][0].cpu().numpy() * (2**16 - 1)

            prob_map = fake_A[j][0].cpu().numpy()
            mask2 = fake_A_thr[j][0].cpu().numpy()
            img2 = fake_B2[j][0].cpu().numpy() * (2**16 - 1)

            if args["plot"]:
            # plot_img_comp(mask, [img])
                plot_img_comp(mask, [img, prob_map, img2], 
                            img_cmap=["gray", "gray", "hot", "gray"], 
                            title=["Simple mask", "Generated image", 
                                    "Probability map", "Generated second image"])
                
            # save images
            if args["save"] or args["save_path"] is not None:
                if args["save_path"] is None:
                    save_path = pathlib.Path("out/tmp/")    
                else:
                    save_path = pathlib.Path(args["save_path"])
                
                if not save_path.is_dir():
                    os.makedirs(save_path)
                    os.mkdir(save_path / "masks")
                    os.mkdir(save_path / "images")
                
                save_img(mask2, path = save_path / "masks" / "{}.png".format(image_counter))
                save_img(img2, path = save_path / "images" /  "{}.png".format(image_counter))
            
            if args["debug"] and i == 3:
                break
            
            image_counter += 1
            
def gen_segmentation_masks(args):     
    # setup data loader
    if "num_workers" in args.keys():
        num_workers = args["num_workers"]
    else:
        num_workers = 2
       
    img_size = args["img_size"]
    batch_size = args["batch_size"]
    base_dir = args["base_dir"]
    
    data_module = CycleGANDataModule(base_dir=base_dir, 
                                     data_dir={"A": "images", "B": "masks"},
                                     img_size= img_size, 
                                     batch_size=batch_size, 
                                     num_workers=num_workers,
                                     norm_mode="MinMax",
                                     norm_params={"min": 0, "max": 2**16 -1},
                                     transforms=None)
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
        
    # create model
    checkpoint_path = args["checkpoint_path"]  
    
    model = CycleGAN.load_from_checkpoint(img_sz=img_size, 
                                          in_channels=1,
                                          checkpoint_path=checkpoint_path)
    
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        model.cuda()
    
    # iterate over the given data
    i = 0
    for batch in tqdm(iter(test_dataloader)):    
        
        with torch.no_grad():
            image = batch["A"]
            if cuda_avail:
                image = image.cuda()
            
            # generate a binary mask based for the generated fake image
            mask_pred, _ = model.forward(image, torch.zeros_like(image))
            mask_pred = torch.where(torch.abs(mask_pred) > 0.9, 1, 0).to(torch.float32)
     
        mask = mask_pred[0][0].cpu().numpy()
        img = image[0][0].cpu().numpy() * (2**16 - 1)

        if args["plot"]:
            plot_img_comp(img, [mask])
        
        # save images
        if args["save"] or args["save_path"] is not None:
            if args["save_path"] is None:
                save_path = pathlib.Path("out/tmp/")    
            else:
                save_path = pathlib.Path(args["save_path"])
            
            if not save_path.is_dir():
                os.makedirs(save_path)
                os.mkdir(save_path / "masks")
                os.mkdir(save_path / "images")
            
            save_img(mask, path = save_path / "masks" / "{}.png".format(i))
            save_img(img, path = save_path / "images" /  "{}.png".format(i))
                    
        i += 1
        if args["debug"] and i == 3:
            break