import numpy as np
import pathlib
import os 
from utils.files import load_raw_img
from utils.plots import plot_img
from binascii import hexlify

if __name__ == "__main__":
    path = pathlib.Path("data\Codierung\proj_Variation_Bildqualität_3_0000.raw")
    path_e = pathlib.Path("data\Codierung\proj_Variation_Bildqualität_3_0000_encoded.raw")
        
    # img = load_raw_img(path, (1984,1984), np.float32, file_offset=2048)
    # plot_img(img)
    
    
    with open(path, "rb") as f:
        # set up the file
        f.seek(2048, os.SEEK_SET)
        
        line = f.readline()
        print(hexlify(line))
        test = bytes(hexlify(line))
        print(hex(test[10]))
    
    # with open(path_e, "r+b") as f:
    #     # set up the file
    #     f.seek(2048, os.SEEK_SET)
            
    #     line = f.readline()
    #     print(hexlify(line))