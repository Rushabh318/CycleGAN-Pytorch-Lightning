import os
import numpy as np
import glob
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import torchvision
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from datasets.cyclegan import CycleGANDataModule
from models.cyclegan import CycleGAN


def gen_images(check_path, imgPath_b):

    # set basic parameter
    img_size = 512
    
    transforms = T.Compose([T.Resize((720, 1280)),
                            T.ToTensor()
                             ])
    
    transform_back = T.ToPILImage()

    # load model
    checkpoint_path =  check_path  
    model = CycleGAN.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                          img_sz=img_size,
                                          in_channels=3)
    
    cuda_avail = torch.cuda.is_available()

    with torch.no_grad():

        if cuda_avail:
            model.cuda()
        
        imgs_b = sorted(glob.glob(imgPath_b+"*.jpg"))

        for i in tqdm(range(len(imgs_b))):  
        
            real_a = torch.zeros(1, 3, img_size, img_size)
            real_b = Image.open(imgs_b[i])
            real_b = transforms(real_b)
            real_b = torch.unsqueeze(real_b, 0)

            if cuda_avail:
                real_a = real_a.cuda()
                real_b = real_b.cuda()

            # generate a fake real images based on the synthetic images
            _ , fake_A = model.forward(real_a, real_b)
            fake_A = torch.squeeze(fake_A, 0)

            fake_A_img = transform_back(fake_A).convert("RGB")
            #fake_A_img.save("/home/au321555/Thesis/images/gan/v3/33/{0:06d}.jpg".format(i))
            fake_A_img.save("/work/au321555/thesis/Data/train_gan_3/coco_data/images/{0:06d}.jpg".format(i))
        print("{0:06d}.png saved".format(len(imgs_b)))

def main():

    check_path = "/home/au321555/Thesis/code/ct-pore-analysis/logs/cyclegan_WZL-run_big/version_1/checkpoints/epoch=49-step=3000000.ckpt"
    imgPath_b = "/work/au321555/thesis/Data/train_big/coco_data/images/"
    #imgPath_b = "/home/au321555/Thesis/images/synthetic/"

    gen_images(check_path, imgPath_b)

if __name__ == "__main__":
    main()
