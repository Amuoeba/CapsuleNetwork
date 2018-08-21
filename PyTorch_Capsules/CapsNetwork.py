import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLayer import ConvLayer
from CapsuleLayer import CapsuleLayer
from DecoderLayer import DecoderLayer

from main import CUDA



class CapsuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.firstConv = ConvLayer()
        self.primeryCapsules = CapsuleLayer()
        self.secondaryCapsules = CapsuleLayer(routing=True,routing_type="Dinamic")
        self.decoderLayer = DecoderLayer()

        self.mseLoss = nn.MSELoss()
        

    def forward(self,data):
        """CapsuleNet forward function
        Args:
            param1: a batch of your data
        Returns:
            Touple[0]: activity vector of last layer capsule
            Touple[1]: reconstruction of the image 
            Touple[2]: a mask of that represents what digit was classified
        """
             
        out = self.secondaryCapsules(self.primeryCapsules(self.firstConv(data)))        
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
        
        capsule_act_len = torch.sqrt((x**2).sum(dim=2,keepdim=True))        

        first = F.relu(m_plus - capsule_act_len).view(batch_size,-1)
        second = F.relu(capsule_act_len - m_minus).view(batch_size,-1)
        
        ml = lable * first + (1.0-lable) * gamma * second
        ml = ml.sum(dim=1).mean()
        
        return ml
    
    def reconstruction_loss(self, x, target):
        # print("Dim of reconstruction:",x.size())
        # print("Dim of target for rec:",target.size())
        loss = self.mseLoss(x.view(x.size(0),-1),target.view(target.size(0),-1))
        # print("MSElosss:",loss)
        # print("!!!!!!!!!!!!1 Reconstruction loss end !!!!!!!!!!!!!!!!!")
        return loss * 0.0005

    