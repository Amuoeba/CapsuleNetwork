import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers.ConvLayer import ConvLayer
from .Layers.CapsuleLayer import CapsuleLayer
from .Layers.DecoderLayer import DecoderLayer



# from main import CUDA



class CapsuleNet(nn.Module):
    def __init__(self,use_cuda=False):
        super().__init__()        
        self.firstConv = ConvLayer()
        self.primeryCapsules = CapsuleLayer()
        self.secondaryCapsules = CapsuleLayer(routing=True,routing_type="Dinamic",use_cuda=use_cuda)
        self.decoderLayer = DecoderLayer(use_cuda=use_cuda)

        self.mseLoss = nn.MSELoss()
        

    def forward(self,data):
        """CapsuleNet forward function
        Args:
            param1: a batch of your data
        Returns:
            Touple[0]: activity vector of last layer capsule
            Touple[1]: reconstruction of the image 
            Touple[2]: a mask that represents what digit was classified
        """
        print("Data size: ",data.size())
        
        conv_out = self.firstConv(data)
        prymary_caps_out = self.primeryCapsules(conv_out)
        out = self.secondaryCapsules(prymary_caps_out)              
        decoded, masked = self.decoderLayer(out)

        return out,decoded,masked

    def loss(self,out,reconst,lable,data):
        rl = self.reconstruction_loss(reconst,data)
        ml = self.margin_loss(out,lable) 
        total_loss = ml + rl
        return total_loss, (ml,rl)
    
    def margin_loss(self, x, lable):
        m_plus = 0.9
        m_minus = 0.1
        gamma = 0.5
        batch_size = x.size(0)
        
        # print("##### X size: {}".format(x.size()))
        # print(x)
        capsule_act_len = torch.sqrt((x**2).sum(dim=2,keepdim=True))
        # print("aCT LEN size: {}".format(capsule_act_len.size()))
        # print(capsule_act_len)

        first = F.relu(m_plus - capsule_act_len).view(batch_size,-1)**2
        second = F.relu(capsule_act_len - m_minus).view(batch_size,-1)**2
        
        ml = lable * first + (1.0-lable) * gamma * second
        # print("Margin loss shape: {}".format(ml.size()))
        # print(ml)
        ml = ml.sum(dim=1).mean()
        # print("Margin loss shape: {}".format(ml.size()))
        # print(ml)
        return ml
    
    def reconstruction_loss(self, x, target):
        # print("Dim of reconstruction:",x.size())
        # print("Dim of target for rec:",target.size())
        loss = self.mseLoss(x.view(x.size(0),-1),target.view(target.size(0),-1))
        # print("MSElosss:",loss)
        # print("!!!!!!!!!!!!1 Reconstruction loss end !!!!!!!!!!!!!!!!!")
        # print("reconst loss: {}".format(loss.size()))
        print(loss)
        return loss * 0.005

    
    def set_collectData(self,value):
        self.secondaryCapsules.collectData = value

    