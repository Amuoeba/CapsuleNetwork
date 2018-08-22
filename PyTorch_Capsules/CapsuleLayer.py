import torch
import torch.nn as nn
import torch.nn.functional as F

# from main import CUDA

class CapsuleLayer(nn.Module):
    def __init__(self, capsule_dim = 8, in_channels = 256, out_channels = 32, ker_size = 9,stride = 2, routing = False, routing_type = "Dinamic",num_itterations = 3, numPrevCaps = 1152, prevCapsDim = 8, numNextCaps = 10, nextCapsDim = 16 ,use_cuda=False):
        super().__init__()
        self.forward_type = None
        self.W = None
        self.use_cuda = use_cuda
        print("CUDA:",self.use_cuda)

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
                print("U2 size:",u[0].size())
                u = torch.stack(u,4)
                print("U2 size:",u.size())
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

                def forward_route(self,x):                    
                    batchSize = x.size(0)
                    x = torch.stack([x]*numNextCaps,dim=2).unsqueeze(4)
                    W = torch.cat([self.W] * batchSize,dim=0)
                    prediction = torch.matmul(W,x)

                    
                    b_ij = torch.zeros(1,numPrevCaps,numNextCaps,1,requires_grad=True)
                    
                    if self.use_cuda:
                        b_ij.cuda()

                    for i in range(num_itterations):
                        c_ij = F.softmax(b_ij,dim=1)
                        if use_cuda:
                            c_ij = c_ij.cuda()

                        c_ij = torch.cat([c_ij] * batchSize, dim=0).unsqueeze(4)


                        s_j = (c_ij * prediction).sum(dim=1,keepdim=True)
                        if self.use_cuda:
                            s_j = s_j.cuda()

                        v_j = self.squash(s_j)



                        if i < num_itterations - 1:
                            a_ij = torch.matmul(prediction.transpose(3,4),torch.cat([v_j] * numPrevCaps, dim = 1))
                            a_ij = a_ij.squeeze(4).mean(dim=0,keepdim=True)
                            print("INSIDE CUDA:",self.use_cuda)
                            if self.use_cuda:
                                a_ij = a_ij.cuda()
                            b_ij = b_ij + a_ij

                    out = v_j.squeeze(1)
                    return out       

                self.forward_type = forward_route
        
    

    def forward(self,x):
        return self.forward_type(self,x)





    def squash(self,capsIn):
        capsSquareNorm = (capsIn**2).sum(-1,keepdim=True)
        capsSum = torch.sqrt(capsSquareNorm)
        capsOut = capsSquareNorm*capsIn/(1+capsSquareNorm*capsSum)
        return capsOut
        

