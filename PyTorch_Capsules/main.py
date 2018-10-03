import torch
import json
import sys
import time
import pandas as pd
import pickle
from itertools import islice
from torch.optim import Adam
import numpy as np
from Networks.MNISTCapsNetwork import CapsuleNet
import utills
import math
from data_readers.mnist_reader import Mnist



with open("./configs/config_MNIST.json") as json_data:
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

# Instanciating the network
caps_net = CapsuleNet(use_cuda=CUDA)
print(caps_net)
if CUDA:
    caps_net.cuda()

model_parameters = filter(lambda p: p.requires_grad, caps_net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("No. parameters: ",params)

# Instantiating the train loader
mnist = Mnist(batch_size,rotate)

# instantiate the optimizer
optimizer = Adam(caps_net.parameters(),lr=learning_rate)

caps_net.train()

best_loss = math.inf

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
        print("Target: {}".format(target_batch.size()))
        print("Image: {}".format(image_batch.size()))
        print("Image: \n")
        print(image_batch)    

        lable = torch.eye(10).index_select(dim=0,index=target_batch)


        if CUDA:
            image_batch = image_batch.cuda()
            lable = lable.cuda()

        optimizer.zero_grad()

        out, decoded, masked = caps_net(image_batch)

        loss = caps_net.loss(out,decoded,lable,image_batch)
        # t0 = time.time()
        loss[0].backward()
        # t1 = time.time()
        # tdiff=t1-t0
        # print("Time Backwards: {}".format(tdiff))
        optimizer.step()


        total_loss = loss[0].data.item()
        margin_loss = loss[1][0].item()
        reconstruction_loss = loss[1][1].item()
        train_loss = train_loss + total_loss
        train_accuracy = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(batch_size)
        
        # print("Epoch",epoch,"Batch:",batch_number)
        # # print("Pred:",np.argmax(masked.data.cpu().numpy(),1))
        # # print("Targ:",np.argmax(lable.data.cpu().numpy(), 1))
        # print("Epoch loss:",train_loss)
        # print("Combined loss:",total_loss)
        # print("Margin loss:", margin_loss, "Reconstruction loss:", reconstruction_loss)
        # print("Train accuracy:", train_accuracy)

        #save model if its combined loss is less than current minimum
        if total_loss < best_loss:
            print("FOUND BETTER MODEL")
            model_to_save = utills.BestModelLog(caps_net.state_dict(),epoch,batch_number,total_loss,margin_loss,reconstruction_loss) 
            exp_env.save_best_model(model_to_save)
            best_loss = total_loss
            
        

        cur_df = pd.DataFrame({"epoch":[int(epoch)],"batch":[int(batch_number)],"margin-loss":[margin_loss],"reconstruction-loss":[reconstruction_loss],"total-loss":[total_loss],"accuracy":[train_accuracy]})
        exp_env.train_data = exp_env.train_data.append(cur_df)        
        
        if batch_number % collection_step == 0:
            all_images = decoded.cpu().detach().numpy()            
            all_coupling_states =np.squeeze(np.array(caps_net.secondaryCapsules.collectedData),axis=0).swapaxes(0,1)
            # print("Shape #########:",all_coupling_states.shape)
            indices = utills.CollectedData.find_first_occurance_index(target_batch)
           

            images = {}
            coupling_states = {}
            for i in range(10):
                if not indices[i] == None:
                    images[i] = all_images[indices[i]]
                    coupling_states[i] = all_coupling_states[:][indices[i]]
                else:
                    images[i] = None
                    coupling_states[i] = None 

            caps_net.secondaryCapsules.collectedData = []
            images = utills.CollectedData("image",images,epoch,batch_number)
            coupling_states = utills.CollectedData("coupling_coefficients",coupling_states,epoch,batch_number)
            exp_env.additional_collected_data.append({"image":images,"coupling":coupling_states})

            exp_env.create_plots(verbose=True)
            exp_env.flush_collected_data()




total_test_loss = 0
no_batches = 0
total_accuracy = 0

with torch.no_grad():

    caps_eval_model = CapsuleNet(use_cuda=CUDA)
    caps_eval_model.load_state_dict(exp_env.best_model)

    if CUDA:
        caps_eval_model.cuda()

    for batch_number, data in islice(enumerate(mnist.test_loader),None,islice_range,None):
        image_batch = data[0]
        
        target_batch = data[1]
        no_examples = image_batch.size(0)
        print("Image batch size: {}".format(image_batch.size()))

        lable = torch.eye(10).index_select(dim=0,index=target_batch)

        if CUDA:
            image_batch = image_batch.cuda()
            lable = lable.cuda()

        out, decoded, masked = caps_eval_model(image_batch)

        loss = caps_eval_model.loss(out,decoded,lable,image_batch)
        total_test_loss += loss[0]
        accuracy = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(lable.data.cpu().numpy(), 1)) / float(no_examples)
        no_batches += 1
        total_accuracy += accuracy        
        print("Batch:",batch_number,"Test accuracy:",accuracy)

print("################################")
print("Final test accuracy:",total_accuracy/no_batches)
print("################################")



