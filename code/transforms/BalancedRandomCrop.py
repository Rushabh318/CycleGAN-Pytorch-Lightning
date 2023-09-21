import torch
from torch import Tensor

from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F

from typing import Tuple

class BalancedRandomCrop(RandomCrop):

    def __init__(self, size, balanced=False, threshold=500, **kwargs):
        super().__init__(size, **kwargs)

        self.balanced = balanced
        self.threshold = threshold

    def compute_choices(self, img):
        patch_sz = self.size

        width, height = F.get_image_size(img)

        num_x_patches = int(width / patch_sz[0]) 
        num_y_patches = int(height / patch_sz[1])

        choices = []

        for i in range(num_x_patches - 1):
            for j in range(num_y_patches - 1):
                dev = torch.std(img[i*patch_sz[0]:(i+1)*patch_sz[0], j*patch_sz[1]:(j+1)*patch_sz[1]])
                if dev > self.threshold:
                    choices.append((i, j))

        return choices


    def get_random_patch(self, img, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if self.balanced:

            #img = F.center_crop(img, (width-width%size[0], height-height%size[1]))

            choices = self.compute_choices(img)
            if len(choices) == 0:
                return self.get_params(img, size)

            rand_idx = torch.randint(len(choices), size=(1, )).item()
            choice = choices[rand_idx]

            i, j, th, tw =  int(choice[0]*size[0]), int(choice[1]*size[1]), int((choice[0]+1)*size[0]), int((choice[1]+1)*size[1])

            return i, j, th, tw

        else:
            return self.get_params(img, size)


    def forward(self, img):
        i, j, h, w = self.get_random_patch(img, self.size)
        return F.crop(img, i, j, h, w)


class _BalancedRandomCrop(RandomCrop):

    def __init__(self, size, balanced=False, threshold=None, **kwargs):
        super().__init__(size, **kwargs)

        self.balanced = balanced
        self.threshold = threshold
        self.max_iter = 32
        self.n = 0
        self.looping = False

    def forward(self, img):

        if self.looping:
            self.n += 1
        
        if self.padding is not None:
            img = pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        upper = self.threshold * h * w
        lower = (1 - self.threshold) * h * w

        if torch.count_nonzero(img[1, ...]) < lower or torch.count_nonzero(img[1, ...]) > upper:
            self.looping = False
            self.n = 0
            return F.crop(img, i, j, h, w)

        if self.balanced:
            cropped = F.crop(img, i, j, h, w)

            if self.n > self.max_iter:
                self.looping = False
                return cropped

            if torch.count_nonzero(cropped[1, ...]) > upper or torch.count_nonzero(cropped[1, ...]) < lower \
                and self.n < self.max_iter :
                    self.looping = True
                    return self.forward(img)
            else:
                self.looping = False
                self.n = 0
                return cropped

        else:
            self.n = 0
            return F.crop(img, i, j, h, w)