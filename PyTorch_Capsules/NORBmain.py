import torch
import json
import sys
import time
import pandas as pd
import pickle
from itertools import islice
from torch.optim import Adam
import numpy as np
from Networks.NORBcapsNet import CapsuleNet
import utills
import math
import data_readers.Small_norb_reader as NORBreader






with open("./configs/config_NORB.json") as json_data:
    config = json.load(json_data)


config = utills.InitConfig.check_config(config,sys.argv)

CUDA = config["cuda"]
islice_range = config["islice"]
batch_size = config["batch size"]
no_epochs = config["epochs"]
collection_step = config["collection step"]
learning_rate = config["lr"]
rotate = config["test_rotate"]
exp_name =config["experiment_name"]

# Data collection, image representations and plotting
exp_env = utills.PrepareExperiment(1,exp_name=exp_name)
# utills.ImagePlotter.plot_NORB_batch_examples(data_loader,4,2)

# Instanciating the network
caps_net = CapsuleNet(use_cuda=CUDA)
print(caps_net)
if CUDA:
    caps_net.cuda()

model_parameters = filter(lambda p: p.requires_grad, caps_net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("No. parameters: ",params)

# Instantiating the train loader
train_data_reader = NORBreader.myNORBreader("../small_NORB_data/",train=True,transform=True)
train_NORB_data_loader = NORBreader.myNORBloader(train_data_reader,batch_size=batch_size,shuffle=True)
test_data_reader = NORBreader.myNORBreader("../small_NORB_data/",train=False,transform=True)
test_NORB_data_loader = NORBreader.myNORBloader(test_data_reader,batch_size=batch_size,shuffle=True)

# instantiate the optimizer
optimizer = Adam(caps_net.parameters(),lr=learning_rate)
best_loss = math.inf


for epoch in range(no_epochs):    
    train_loss = 0
    print("Epoch:",epoch)

    for batch_number, data in islice(enumerate(train_NORB_data_loader),None,islice_range,None):

        image_batch = data["image"]
        image_batch = image_batch.unsqueeze(1)
        image_batch = image_batch.type(torch.FloatTensor)
        
        target_batch = data["tag"]
        target_batch = target_batch.type(torch.LongTensor)
        print("Target: {}".format(target_batch.size()))
        print("Image: {}".format(image_batch.size()))        

        lable = torch.eye(5).index_select(dim=0,index=target_batch)
        print(lable.size())

        if CUDA:
            image_batch = image_batch.cuda()
            lable = lable.cuda()

        optimizer.zero_grad()

        out, mask = caps_net(image_batch)

        loss = caps_net.loss(out,lable)
        print(loss)
        # t0 = time.time()
        loss.backward()
        # t1 = time.time()
        # tdiff=t1-t0
        # print("Time Backwards: {}".format(tdiff))
        optimizer.step()

        episode_loss = loss.item()
        train_loss += episode_loss
        # print("Lables: \n",lable)
        # print("Mask: \n",mask)
        train_accuracy = sum(np.argmax(mask.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size)
        print("Train accuracy: {}".format(train_accuracy))
        #save model if its combined loss is less than current minimum
        if episode_loss < best_loss:
            print("FOUND BETTER MODEL")
            model_to_save = utills.BestModelLog(caps_net.state_dict(),epoch,batch_number,episode_loss,None,None) 
            exp_env.save_best_model(model_to_save)
            best_loss = episode_loss

        cur_df = pd.DataFrame({"epoch":[int(epoch)],"batch":[int(batch_number)],"margin-loss":[0],"reconstruction-loss":[0],"total-loss":[episode_loss],"accuracy":[train_accuracy]})
        exp_env.train_data = exp_env.train_data.append(cur_df)

        if batch_number % collection_step == 0:
            exp_env.create_plots(verbose=True)
        
        
        