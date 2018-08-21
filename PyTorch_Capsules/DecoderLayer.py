import torch 
import torch.nn as nn
import torch.nn.functional as F
from utills import ImagePlotter


class DecoderLayer(nn.Module):
    """
    Fully connected decoder layer. Consists of 3 fully conected Linear layer.
    Between all layers a ReLu nonlinearity is used.
    Args:
        l1_in: number of inputs to the first linear layer
        l1_out: number of outputs of the first linear layer
        l2_out: number of outputs of the second linear layer
        l3_out: number of outputs of the third linear layer
    Forward output:
        touple[0]:
            a tensor that represents the recunstructed image
        touple[1]:
            a tensor of masks that represents what was the result 
            of classification
    """
    def __init__(self,l1_in = 160, l1_out=512, l2_out = 1024, l3_out = 784):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.Linear(l1_in,l1_out),
            nn.ReLU(inplace=True),
            nn.Linear(l1_out,l2_out),
            nn.ReLU(inplace=True),
            nn.Linear(l2_out,l3_out),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        # max_class is calculated by checking the lenght of the capsule output vectors.
        # longer vectors mean greater probability for that class to be present
        # print("0---------------",x.size())
        max_class = torch.sqrt(x**2).sum(2)
        # print("1---------------",max_class.size())
        max_class = F.softmax(max_class,dim=1)
        # print("2---------------",max_class.size())
        max_class_indices = max_class.max(dim=1)[1]
        # print("3---------------",max_class_indices.size())
        # print(max_class_indices)
        mask = torch.eye(10)
        # print("4---------------",mask.size())
        mask = mask.index_select(dim=0, index=max_class_indices.squeeze(1).data)
        # print(mask)
        # print("5---------------",mask.size())
        masked_x = x * mask[:,:,None,None]
        # print(masked_x)
        # print("6---------------",masked_x.size())
        masked_x = masked_x.view(x.size(0),-1)
        # print("7---------------",masked_x.size())   
        out = self.decoder_layers(masked_x)
        out = out.view(-1,1,28,28)
        # print(out)

        # plotter = ImagePlotter(destination="./DecoderTest/")
        # to_plot = out[:2,0].detach().numpy()            
        # plotter.plot_images_separately(to_plot,save=True)

        return out, mask