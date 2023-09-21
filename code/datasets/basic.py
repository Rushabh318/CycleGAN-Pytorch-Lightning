import numpy as np
import os
from skimage import io
from tqdm import tqdm

from utils.files import load_img
import torch
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    
    def __init__(self, 
                 datasetA: str = None, 
                 datasetB: str = None,
                 recursive: bool = False,
                 exclude_tags: list = [],
                 base_dir = None,
                 drop: bool = False,
                 norm_mode: str = None,
                 norm_params: dict = None,
                 transforms = None,
                 max_sz: int = None, 
                 shuffle: bool = False,
                 seed: int = 42,
                 **kwargs
                 ):
        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """

        super().__init__()
                
        if datasetA is None or datasetB is None:
            raise KeyError("One or both datasets are not found!")
        
        # TODO: find a better way than hard-coding 
        sort_key = lambda x: int(x.split(".")[0].split("_")[-1])
        
        # Search base path for datasets
        if recursive and base_dir is not None:

            self.datasetA_names = []
            self.datasetB_names = []

            for root, dirnames, _ in os.walk(base_dir):
                if not any([tag in root for tag in exclude_tags]):
                    for dirname in dirnames:
                        if dirname == datasetA:
                            filenames = os.listdir(os.path.join(root, dirname))
                            self.datasetA_names.extend([os.path.join(root, dirname, name) for name in filenames])
                        elif dirname == datasetB:
                            filenames = os.listdir(os.path.join(root, dirname))
                            self.datasetB_names.extend([os.path.join(root, dirname, name) for name in filenames])
                        else:
                            continue
                else:
                    continue

            #if max_sz is not None:
            #    self.datasetA_names = self.datasetA_names[:max_sz]
            #    self.datasetB_names = self.datasetB_names[:max_sz]

        else:
            # get images from datasetA        
            datasetA_names = os.listdir(datasetA)    
            self.datasetA_names = [datasetA / name for name in datasetA_names]
            #if max_sz is not None:
            #    self.datasetA_names = self.datasetA_names[:max_sz]

            # get image from datasetB
            datasetB_names = os.listdir(datasetB)
            self.datasetB_names = [datasetB / name for name in datasetB_names]
            #if max_sz is not None:
            #    self.datasetB_names = self.datasetB_names[:max_sz]

        # drop images with low std dev
        if drop:
            self.drop_std_dev_below(500)

        # shuffle
        if shuffle:
            np.random.seed(seed)
            names = list(zip(self.datasetA_names, self.datasetB_names))
            np.random.shuffle(names)
            self.datasetA_names, self.datasetB_names = zip(*names) 

        # crop to max_sz
        if max_sz is not None:
            self.datasetA_names = self.datasetA_names[:max_sz]
            self.datasetB_names = self.datasetB_names[:max_sz]

        self.norm_mode = norm_mode
        self.norm_params = norm_params

        # transform together (e.g. for image/masks)
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = None # T.Compose([T.ToTensor()])

        self.kwargs = kwargs
      
    def __len__(self):
        return min(len(self.datasetA_names), len(self.datasetB_names))


    def __getitem__(self, idx):
        # each sample has instances of both datasets A and B
        sample = {}

        # load instance from A 
        imgA = Image.open(self.datasetA_names[idx % len(self.datasetA_names)])      
                      
        # load instance from B
        imgB = Image.open(self.datasetB_names[idx % len(self.datasetB_names)])

        # perform the same transformation on the sample parts
        if self.transforms is not None:
            # apply seperate transforms to both datasets
            if type(self.transforms) == dict:
                if not all(key in ["A", "B"] for key in self.transforms.keys()):
                    raise KeyError
                sample["A"] = self.transforms["A"](imgA)
                sample["B"] = self.transforms["B"](imgB)
            # otherwise, apply the same transform
            else:
                imgA = self.transforms(imgA)
                imgB = self.transforms(imgB)

                imgA = np.float32(imgA)/255.0
                imgB = np.float32(imgB)/255.0
                imgA = torch.from_numpy(imgA)
                imgB = torch.from_numpy(imgB)
                imgA = torch.movedim(imgA, 2, 0)
                imgB = torch.movedim(imgB, 2, 0)

                sample["A"] = imgA
                sample["B"] = imgB
        else:
            sample["A"] = imgA
            sample["B"] = imgB

        # modify target type (imgB) if necessary 
        if "target_type" in self.kwargs.keys():
             # one hot encoding
            if self.kwargs["target_type"] == "one_hot":
                if "num_classes" not in self.kwargs.keys():
                    raise KeyError("Number of classes missing!")
                
                imgB = torch.squeeze(sample["B"], 0)
                
                imgB = torch.nn.functional.one_hot(imgB.long(), self.kwargs["num_classes"])
                imgB = torch.swapaxes(imgB, 0, 2)
                imgB = torch.swapaxes(imgB, 1, 2)

                sample["B"] = imgB.int()
            
            elif self.kwargs["target_type"] == "class_matrix":
                if "num_classes" not in self.kwargs.keys():
                    raise KeyError("Number of classes missing!")

                imgB = sample["B"]
                imgB = imgB/255.0 * self.kwargs["num_classes"]
                if self.kwargs["num_classes"] == 1:
                    imgB[imgB != 0] = 1
                sample["B"] = imgB.float()
                pass

            else:
                raise KeyError()
            
        return sample

    def normalize(self, img):
        if self.norm_mode == "MinMax":
            if "min" not in self.norm_params.keys() or "max" not in self.norm_params.keys():
                raise KeyError()
            
            img = (img - self.norm_params["min"]) / (self.norm_params["max"] - self.norm_params["min"])
        elif self.norm_mode == "PerImgMinMax":
            
            img_min = np.min(img)
            img_max = np.max(img)

            img = (img - img_min) / (img_max - img_min)
        else:
            raise NotImplementedError()
        
        return img

    def drop_std_dev_below(self, threshold=500):
        remove_indices = []
        for i in tqdm(range(len(self))):
            if np.std(io.imread(self.datasetA_names[i])) < threshold:
                remove_indices.append(i)
        for idx in sorted(remove_indices, reverse=True):
            del self.datasetA_names[idx]
            del self.datasetB_names[idx]

class DoubleDataset(Dataset):
    
    def __init__(self, 
                 datasetA: str = None, 
                 datasetB: str = None,
                 transfA = None, 
                 transfB = None,
                 max_sz: int = 1000, 
                 ):
        """
        Parameters:
            transforms: a list of Transformations (Data augmentation)
        """

        super().__init__()
                
        # check whether datasets have been given
        if datasetA is None or datasetB is None:
            raise KeyError("One or both datasets are not found!")
        
        # get images from datasetA
        datasetA_names = os.listdir(datasetA)            
        self.datasetA_names = [datasetA + "/" + name for name in datasetA_names]
        self.datasetA_names = self.datasetA_names[:max_sz]
        
        self.transfA = transfA
        
        # get image from datasetB
        datasetB_names = os.listdir(datasetB)
        self.datasetB_names = [datasetB + "/" + name for name in datasetB_names]
        self.datasetB_names = self.datasetB_names[:max_sz]
        
        self.transfB = transfB   
      
    
    def __len__(self):
        return min(len(self.datasetA_names), len(self.datasetB_names))


    def __getitem__(self, idx):
        # each sample has instances of both datasets A and B
        sample = {}   
        
        # get instance from A 
        imgA = io.imread(self.datasetA_names[idx % len(self.datasetA_names)])
        imgA = np.float32(imgA)
        
        # the img has to be of dim (h,w,c) even if gray-values are given
        if imgA.ndim == 2:
            imgA = np.expand_dims(imgA, -1)          
        if self.transfA is not None:
            sample["A"] = self.transfA(imgA)
         
        # get instance from B
        imgB =  io.imread(self.datasetB_names[idx % len(self.datasetB_names)])
        imgB = np.float32(imgB)
        
        if imgB.ndim == 2:
            imgB = np.expand_dims(imgB, -1)
        if self.transfB is not None:
            sample["B"] = self.transfB(imgB)

        return sample
