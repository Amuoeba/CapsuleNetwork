from torchvision import transforms, utils
import torch
import numpy as np
from torchvision import transforms, utils

class NORBtransform(object):
    """
    Transforms a numpy array of an image to a tensor
    """
    def __call__(self, sample):
        image, label = sample["image"], sample["tag"]

        image = np.expand_dims(image, axis=3)

        image = transforms.ToTensor()(image)

        # image = transforms.Normalize((0.1307,), (0.3081,))(image)

        return {"image": image, "tag": label}
