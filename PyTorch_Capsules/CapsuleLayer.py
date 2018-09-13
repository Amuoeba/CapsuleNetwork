import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# from main import CUDA

class CapsuleLayer(nn.Module):
    def __init__(self, capsule_dim = 8, in_channels = 256, out_channels = 32, ker_size = 9,stride = 2, routing = False, routing_type = "Dinamic",num_itterations = 3, numPrevCaps = 1152, prevCapsDim = 8, numNextCaps = 10, nextCapsDim = 16 ,use_cuda=False):
        super().__init__()
        self.forward_type = None
        self.W = None
        self.use_cuda = use_cuda
        self.collectData = False
        self.collectedData = []
        # print("CUDA:",self.use_cuda)˝

        #Check whether routing should be conducted or not
        if not routing:
            # Number 2D Convolutional models coresponds to the dimension of a capsule output, 8 in our case
            # How many different capsule types coresponds to the number of out_channels, 32 in our case
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ker_size, stride=stride, padding=0)
                for _ in range(capsule_dim)
            ])

            # In the forward pass:
            #   - calculate 2D convolutions on each input x (8 different convolutions with 32 output features in our case)
            #   - We stack the calculated tensors in dimension 1, which essentially means stacking scalar outputs of convolutions to produce 8 dimensional vector outputs
            #   - Transform the tensor so that we only have individual 8D capsules (32*6*6 = 1152) in each batch --> [batch,1152,8]
            def forward_no_route(self,x):
                u = [capsule(x) for capsule in self.capsules]
                # print("U2 size:",u[0].size())
                u = torch.stack(u,4)
                # print("U3 size:",u.size())
                u = u.view(x.size(0),32*6*6,-1)

                assert u.size() == torch.Size([x.size(0),1152,8])
                return self.squash(u)

            self.forward_type = forward_no_route


        elif routing:
            if routing_type == "Dinamic":
                numPrevCaps = numPrevCaps
                prevCapsDim = prevCapsDim
                numNextCaps = numNextCaps
                nextCapsDim = nextCapsDim

                self.W = nn.Parameter(torch.randn(1,numPrevCaps,numNextCaps,nextCapsDim,prevCapsDim))
                self.biases = nn.Parameter(torch.zeros(1,numPrevCaps,numNextCaps,nextCapsDim,1))

                

                def forward_route(self,x):                    
                    batchSize = x.size(0)
                    # print("X before: {}".format(x.size()))
                    x = torch.stack([x]*numNextCaps,dim=2).unsqueeze(4)
                    W = torch.cat([self.W] * batchSize,dim=0)
                    # print("X dim: {}".format(x.size()))
                    # print("W dim: {}".format(W.size()))                    
                    prediction = torch.matmul(W,x) + self.biases
                    # print("x: {}".format(x.size()))
                    # print("W: {}".format(W.size()))
                    # print("Prediction: {}".format(prediction.size()))

                    # print(self.biases)
                    
                    b_ij = torch.zeros(batchSize,numPrevCaps,numNextCaps,1,requires_grad=False)
                    pred_nograd = torch.tensor(prediction,requires_grad = False)
                    
                    if self.use_cuda:
                        b_ij.cuda()
                    
                    if self.collectData:
                        colledtion = []
                    # print("---------- ROUTING START ----------")
                    for i in range(num_itterations):
                        # print("Itteration: {} ,b_ij Size: {}".format(i,b_ij.size()))
                        c_ij = F.softmax(b_ij,dim=2)
                        # print(c_ij)
                        # print("C_ij Size: {}".format(c_ij.size()))
                        
                        
                        if use_cuda:
                            c_ij = c_ij.cuda()

                        #c_ij = torch.cat([c_ij] * batchSize, dim=0).unsqueeze(4)
                        c_ij = torch.tensor(c_ij).unsqueeze(4)
                        # print("C_ij Size adter unsqueeze: {}".format(c_ij.size()))
                        
                        

                        if self.collectData:
                            c_analize = torch.tensor(c_ij).cpu().squeeze().detach().numpy()
                            c_analize = np.reshape(c_analize,(batchSize,10,32,6,-1))
                            # print("C_analize shape: {}".format(c_analize.shape))
                            colledtion.append(c_analize)             

                        


                        # print("V_j: {}".format(v_j.size()))
                        # print(v_j)
                        if i < num_itterations - 1:
                            # print(c_ij)
                            # print(pred_nograd)
                            # print("Product size: {}".format((c_ij * pred_nograd).size()))
                            s_j = (c_ij * pred_nograd).sum(dim=1,keepdim=True)
                            # print(s_j)
                            # print("S_j: {}".format(s_j.size()))
                            # print(s_j)
                            if self.use_cuda:
                                s_j = s_j.cuda()

                            v_j = self.squash(s_j)
                            if self.use_cuda:
                                v_j = v_j.cuda()
                            # print("V j: {}".format(v_j.size()))
                            # print("Prediction: {}".format(pred_nograd.size()))
                            # print("Prediction transpose: {}".format(pred_nograd.transpose(3,4).size()))
                            
                            a_ij = torch.matmul(pred_nograd.transpose(3,4),torch.cat([v_j] * numPrevCaps, dim = 1))
                            # print("A_ij: {}".format(a_ij.size()))
                            # print(a_ij)
                            # a_ij = a_ij.squeeze(4).mean(dim=0,keepdim=True)
                            a_ij = a_ij.squeeze(4)
                            # print("B ij size : {}".format(b_ij.size()))
                            
                            if self.use_cuda:
                                a_ij = a_ij.cuda()
                            if self.use_cuda:
                                b_ij = b_ij.cuda()
                            b_ij = b_ij + a_ij

                        elif i == num_itterations -1:
                            s_j = (c_ij * prediction).sum(dim=1,keepdim=True)
                            # print("S_j: {}".format(s_j.size()))
                            if self.use_cuda:
                                s_j = s_j.cuda()

                            v_j = self.squash(s_j)
                            if self.use_cuda:
                                v_j = v_j.cuda()
                        
                    
                    if self.collectData:
                        self.collectedData.append(colledtion)


                    out = v_j.squeeze(1)
                    return out       

                self.forward_type = forward_route
        
    

    def forward(self,x):
        return self.forward_type(self,x)





    def squash(self,capsIn):
        #print("CAps in : {}".format(capsIn.size()))
        capsSquareNorm = (capsIn**2).sum(-2,keepdim=True)
        capsSum = torch.sqrt(capsSquareNorm)
        capsOut = capsSquareNorm*capsIn/(1+capsSquareNorm*(capsSum + 1e-9))
        return capsOut
        

