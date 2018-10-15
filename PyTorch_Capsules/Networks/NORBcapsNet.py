import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers.ConvLayer import ConvLayer
from .Layers.CapsuleLayer import CapsuleLayer
from .Layers.DecoderLayer import DecoderLayer


# from main import CUDA


class CapsuleNet(nn.Module):
    def __init__(self, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda
        self.conv1 = ConvLayer(1, 64, 5, 1)
        self.conv2 = ConvLayer(64, 64, 6, 2)
        self.conv3 = ConvLayer(64, 256, 6, 2)
        self.primeryCapsules = CapsuleLayer()
        self.secondaryCapsules = CapsuleLayer(routing=True, routing_type="Dinamic", use_cuda=use_cuda, numNextCaps=5, num_itterations=3)
        # self.decoderLayer = DecoderLayer(use_cuda=use_cuda)
        self.mseLoss = nn.MSELoss()

    def forward(self, data):
        """CapsuleNet forward function
        Args:
            param1: a batch of your data
        Returns:
            Touple[0]: activity vector of last layer capsule
            Touple[1]: reconstruction of the image 
            Touple[2]: a mask that represents what digit was classified
        """
        print("Data size: ", data.size())

        conv1_out = self.conv1(data)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        # print("Conv3 out size: {}".format(conv3_out.size())) 
        prymary_caps_out = self.primeryCapsules(conv3_out)
        out = self.secondaryCapsules(prymary_caps_out)
        # print("Sec Caps out size: {}".format(out.size()))
        # print("Out: \n",out)
        mask = self.select_max_class(out)
        # print("Prediction size: {}".format(mask.size()))
        # print("Prediction: \n",mask)
        # decoded, masked = self.decoderLayer(out)
        return out, mask

    def loss(self, out, lable):  # reconst,lable,data):
        # rl = self.reconstruction_loss(reconst,data)
        ml = self.margin_loss(out, lable)
        # total_loss = ml + rl
        return ml
        # return total_loss, (ml,rl)

    def margin_loss(self, x, lable):
        m_plus = 0.9
        m_minus = 0.1
        gamma = 0.5
        batch_size = x.size(0)

        # print("##### X size: {}".format(x.size()))
        # print(x)
        capsule_act_len = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        # print("aCT LEN size: {}".format(capsule_act_len.size()))
        # print(capsule_act_len)

        first = F.relu(m_plus - capsule_act_len).view(batch_size, -1) ** 2
        second = F.relu(capsule_act_len - m_minus).view(batch_size, -1) ** 2

        ml = lable * first + (1.0 - lable) * gamma * second
        # print("Margin loss shape: {}".format(ml.size()))
        # print(ml)
        ml = ml.sum(dim=1).mean()
        # print("Margin loss shape: {}".format(ml.size()))
        # print(ml)
        return ml

    # def reconstruction_loss(self, x, target):
    #     # print("Dim of reconstruction:",x.size())
    #     # print("Dim of target for rec:",target.size())
    #     loss = self.mseLoss(x.view(x.size(0),-1),target.view(target.size(0),-1))
    #     # print("MSElosss:",loss)
    #     # print("!!!!!!!!!!!!1 Reconstruction loss end !!!!!!!!!!!!!!!!!")
    #     # print("reconst loss: {}".format(loss.size()))
    #     print(loss)
    #     return loss * 0.005

    def select_max_class(self, x):
        max_class = torch.sqrt(x ** 2).sum(2)
        max_class = F.softmax(max_class, dim=1)
        max_class_indices = max_class.max(dim=1)[1]

        mask = torch.eye(5)
        if self.use_cuda:
            mask = mask.cuda()
        mask = mask.index_select(dim=0, index=max_class_indices.squeeze(1).data)
        return mask

    def set_collectData(self, value):
        self.secondaryCapsules.collectData = value
