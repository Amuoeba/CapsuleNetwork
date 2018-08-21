import torch 
import torch.nn as nn
import torch.nn.functional as F

from main import CUDA

class ConvLayer(nn.Module):
    """
    ConvLayer is a clasic convolution layer
    Args:
        in_chan: number of input channels. Default = 1
        out_chan: number of output channels. Default = 256
        ker_size: kernel size. Default = 9
        stride: stride. Default = 1
    Forward output:
        Result of a 2D convolution
    """
    def __init__(self, in_chan = 1, out_chan = 256, ker_size = 9, stride = 1):
        super().__init__()
        self.convolution = nn.Conv2d(in_chan, out_chan, ker_size, stride)
    
    def forward(self,input_tensor):
        out = F.relu(self.convolution(input_tensor))
        # print("Size of the FIRST conv output:",out.size())
        return out

