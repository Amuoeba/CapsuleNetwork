from torchvision import transforms, utils
import torch
import numpy as np

class NORB_ToTensor(object):
    """
    Transforms a numpy array of an image to a tensor
    """
    def __call__(self,sample):
        image, landmarks = sample["image"], sample["tag"]
        print("Image size {}".format(image.size()))
        print("Lable size {}".format(lable))
