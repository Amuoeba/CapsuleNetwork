import torch
from CapsNetwork import CapsuleNet
from utills import ImagePlotter
from data_reader import Mnist
from torch.optim import Adam
import numpy as np
from itertools import islice
import time


# #Inptu is of shape [10,1,28,28] -> [batch,chan,dim_x,dim_y]
# print("Size of the DATA:",a[0].size())
# plot_images_separately(a[0][:10, 0].data.cpu().numpy())


# CUDA
CUDA = True

# Plotter for ploting the raw images and reconstructions
plotter = ImagePlotter()

# Instanciating the network
caps_net = CapsuleNet(use_cuda=CUDA)
if CUDA:
    caps_net.cuda()

model_parameters = filter(lambda p: p.requires_grad, caps_net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("No. parameters: ",params)


# Training parameters
no_epochs = 30
batch_size = 100

# Instantiating the train loader
mnist = Mnist(batch_size)


# instantiate the optimizer
optimizer = Adam(caps_net.parameters())

caps_net.train()
for epoch in range(no_epochs):
    
    train_loss = 0
    print("Epoch:",epoch)
    for batch_number, data in enumerate(mnist.train_loader): #islice(generator,to,step)

        image_batch = data[0]
        target_batch = data[1]      

        lable = torch.eye(10).index_select(dim=0,index=target_batch)        

        if CUDA:
            image_batch = image_batch.cuda()
            lable = lable.cuda()

        optimizer.zero_grad()

        out, decoded, masked = caps_net(image_batch)

        loss = caps_net.loss(out,decoded,lable,image_batch)
        loss[0].backward()

        optimizer.step()
        train_loss = train_loss + loss[0].data.item()       
        print("Epoch",epoch,"Batch:",batch_number)
        # print("Pred:",np.argmax(masked.data.cpu().numpy(),1))
        # print("Targ:",np.argmax(lable.data.cpu().numpy(), 1))
        print("Train loss",train_loss)
        print("Margin loss:", loss[1][0], "Reconstruction loss:", loss[1][1])
        print("Train accuracy:",sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size))
        
        if batch_number % 10 == 0:
            # print("Decoded size",decoded.size())
            # print("Train accuracy:",sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size))
            # print("Num examples:",decoded[:10,0].size())
            name = "image_" + str(epoch) + "_"
            to_plot = decoded[:10,0].cpu().detach().numpy()            
            plotter.plot_images_separately(to_plot,save=True,name=name)


time.sleep(1)


total_test_loss = 0


# test_mnist = Mnist(10)
# print(next(enumerate(test_mnist.test_loader)))

with torch.no_grad():
    for batch_number, data in enumerate(mnist.test_loader):
        image_batch = data[0]
        target_batch = data[1]

        lable = torch.eye(10).index_select(dim=0,index=target_batch)

        if CUDA:
            image_batch = image_batch.cuda()
            lable = lable.cuda()

        out, decoded, masked = caps_net(image_batch)

        loss = caps_net.loss(out,decoded,lable,image_batch)
        total_test_loss += loss[0]

        print("Batch:",batch_number,"Test accuracy:",sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size))