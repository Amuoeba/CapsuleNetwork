import torch
import json
import sys
import time
import pandas as pd
import pickle
from itertools import islice
from torch.optim import Adam
import numpy as np
from CapsNetwork import CapsuleNet
import utills
from data_reader import Mnist



with open("./config.json") as json_data:
    config = json.load(json_data)


config = utills.InitConfig.check_config(config,sys.argv)



CUDA = config["cuda"]
islice_range = config["islice"]
batch_size = config["batch size"]
no_epochs = config["epochs"]
collection_step = config["collection step"]


# #Inptu is of shape [10,1,28,28] -> [batch,chan,dim_x,dim_y]
# print("Size of the DATA:",a[0].size())
# plot_images_separately(a[0][:10, 0].data.cpu().numpy())


# CUDA


# Data collection, image representations and plotting
exp_env = utills.PrepareExperiment(1)

# Instanciating the network
caps_net = CapsuleNet(use_cuda=CUDA)
print(caps_net)
if CUDA:
    caps_net.cuda()

model_parameters = filter(lambda p: p.requires_grad, caps_net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("No. parameters: ",params)


# Training parameters


# Instantiating the train loader
mnist = Mnist(batch_size)


# instantiate the optimizer
optimizer = Adam(caps_net.parameters(),lr=0.0001)

caps_net.train()
for epoch in range(no_epochs):    
    train_loss = 0
    print("Epoch:",epoch)

    for batch_number, data in islice(enumerate(mnist.train_loader),None,islice_range,None):
        if batch_number % collection_step == 0:
            caps_net.set_collectData(True)
        else:
            caps_net.set_collectData(False)

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


        total_loss = loss[0].data.item()
        margin_loss = loss[1][0].item()
        reconstruction_loss = loss[1][1].item()
        train_loss = train_loss + total_loss
        train_accuracy = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size)
        
        print("Epoch",epoch,"Batch:",batch_number)
        # print("Pred:",np.argmax(masked.data.cpu().numpy(),1))
        # print("Targ:",np.argmax(lable.data.cpu().numpy(), 1))
        print("Train loss",train_loss)
        print("Margin loss:", margin_loss, "Reconstruction loss:", reconstruction_loss)
        print("Train accuracy:", train_accuracy)

        cur_df = pd.DataFrame({"epoch":[int(epoch)],"batch":[int(batch_number)],"margin-loss":[margin_loss],"reconstruction-loss":[reconstruction_loss],"total-loss":[total_loss],"accuracy":[train_accuracy]})
        exp_env.train_data = exp_env.train_data.append(cur_df)        
        
        if batch_number % collection_step == 0:            
            images = decoded[:10,0].cpu().detach().numpy()
            coupling_states = caps_net.secondaryCapsules.collectedData
            caps_net.secondaryCapsules.collectedData = []

            images = utills.CollectedData("image",images,epoch,batch_number)
            coupling_states = utills.CollectedData("coupling_coefficients",coupling_states,epoch,batch_number)
            exp_env.additional_collected_data.append({"image":images,"coupling":coupling_states})




total_test_loss = 0
no_examples = 0
total_accuracy = 0
with torch.no_grad():
    for batch_number, data in islice(enumerate(mnist.test_loader),None,islice_range,None):
        image_batch = data[0]
        target_batch = data[1]

        lable = torch.eye(10).index_select(dim=0,index=target_batch)

        if CUDA:
            image_batch = image_batch.cuda()
            lable = lable.cuda()

        out, decoded, masked = caps_net(image_batch)

        loss = caps_net.loss(out,decoded,lable,image_batch)
        total_test_loss += loss[0]
        accuracy = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size)
        no_examples += batch_size
        total_accuracy += accuracy        
        print("Batch:",batch_number,"Test accuracy:",accuracy)

print("################################")
print("Final test accuracy:",total_accuracy/no_examples)
print("################################")


exp_env.create_plots()

# print(len(exp_env.additional_collected_data))
# print(exp_env.additional_collected_data[0]['coupling'])
# print(np.array(exp_env.additional_collected_data[0]['coupling'].data).shape)