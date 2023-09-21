import numpy as np
import pathlib
from skimage.filters import sobel, laplace
import glob
from tqdm import tqdm

from utils.files import load_raw_img, load_img
from utils.plots import *
from analysis.slices import *

def show_img(data_path, ind, file_format):
    file_list = list(data_path.glob("*.raw"))
    
    img = load_raw_img(file_list[ind], file_format, 
                       file_type=np.uint16)

    plot_img(img, img_cmap="hot", cbar=True)
    
if __name__ == "__main__":
    data_path = pathlib.Path("data/cube/XY_Slices")
    
    for i in range(811, 850):
        show_img(data_path, i, (2464, 1837))