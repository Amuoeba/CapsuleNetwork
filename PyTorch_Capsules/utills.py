import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import os
import sys
import itertools


plt.switch_backend('agg')

class PrepareExperiment():
    def __init__(self, id, home=".",):
        self.id = id
        self.home=home        
        self.expHome = None
        self.image_dest = None
        self.reconst_image_dest = None
        self.coupling_image_dest = None
        self.plots = None
        self._prepare_foldiers()
        self.train_data = pd.DataFrame({"epoch":[],"batch":[],"margin-loss":[],"reconstruction-loss":[],"total-loss":[],"accuracy":[]})
        self.additional_collected_data = []

        self.plotter = ImagePlotter(self.image_dest)
        
    
    def _prepare_foldiers(self):
        dirname = self.unique_name(self.home + "/experiments/experiment",makedir=True)
        img=dirname+"/images"
        img_reconst = "/reconstructions"
        img_coupling = "/coupling"
        
        plots = dirname+"/plots"       
        os.makedirs(dirname)
        os.makedirs(img)   
        os.makedirs(img+img_reconst)
        os.makedirs(img+img_coupling)
        os.makedirs(plots)
        
        self.expHome = dirname+"/"
        self.reconst_image_dest = img_reconst+"/"
        self.coupling_image_dest = img_coupling+"/"
        self.plots = plots+"/"
        self.image_dest = img+"/" 

    @staticmethod
    def unique_name(basename, ext=None, makedir=False):
        c = itertools.count()
        if makedir:
            actualname = "%s" % basename
            while os.path.exists(actualname):
                actualname = "%s_%d" % (basename, next(c))
        else:
            actualname = "%s.%s" % (basename, ext)    
            while os.path.exists(actualname):
                actualname = "%s_%d.%s" % (basename, next(c),ext)
        return actualname
    
    def create_plots(self):
        for data in self.additional_collected_data:
            image = data["image"]
            coupling = data["coupling"]            

            reconst_image_name = "image_" + str(image.epoch) + "_"
            self.plotter.plot_reconstruction_images(image,save=True,name=reconst_image_name,subdest=self.reconst_image_dest)
            couple_image_name = "coupl_" + str(coupling.epoch) + "_BA" + str(coupling.batch)
            self.plotter.plot_coupling_image(coupling,save=True,name=couple_image_name,subdest=self.coupling_image_dest)

        
        self.plot_train_data()



    def plot_train_data(self):
        train_data = self.train_data
        cols = ["epoch","batch"]
        train_data[cols] = train_data[cols].applymap(np.int64)       

        accuracy_graph = sb.relplot(x="batch",y="accuracy",hue="epoch",data=train_data,kind="line")
        totloss_graph = sb.relplot(x="batch",y="total-loss",hue="epoch",data=train_data,kind="line")
        margin_graph = sb.relplot(x="batch",y="margin-loss",hue="epoch",data=train_data,kind="line")
        reconstruction_graph = sb.relplot(x="batch",y="reconstruction-loss",hue="epoch",data=train_data,kind="line")

        accuracy_graph.savefig(self.plots+"accuracy_plot.png")
        totloss_graph.savefig(self.plots+"total_plot.png")
        margin_graph.savefig(self.plots+"margin_plot.png")
        reconstruction_graph.savefig(self.plots+"reconst_plot.png")

    




class ImagePlotter():
    def __init__(self,destination="./images/",name="image_"):
        self.destination = destination
        self.current_reconst = 0
        self.current_coupling = 0
        self.name = name    

    def plot_reconstruction_images(self,images,save=False,name="default",subdest=""):
        "Plot the six MNIST images separately."      
        
        images = images.data
        fig = plt.figure()

        for j in range(10):            
            if not images[j] is None:
                img_data =np.squeeze(images[j])
            else:
                img_data = np.zeros((28,28))
            
            ax = fig.add_subplot(1, 10, j+1)
            ax.matshow(img_data, cmap = matplotlib.cm.binary)
            ax.set_title("#{}".format(j))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        
        if save:
            # print(self.destination)
            # print(subdest)
            # print(name)
            fig.savefig(self.destination + subdest + name + str(self.current_reconst) + ".png")
            self.current_reconst += 1
        else:
            plt.show()
    
    def plot_coupling_image(self,data,save=False,name="default",subdest=""):
        all_data = data.data
        for i in range(10):
            if not all_data[i] is None:
                pltData =  np.array(all_data[i])          
                plt.figure(figsize=(30,30))

                range_min = np.amin(pltData)
                range_max = np.amax(pltData)
                Y_max = pltData.shape[2]
                Y_min = 0
                X_max = pltData.shape[1]
                X_min = 0

                loffset = 4/X_max
                koffset = 2/Y_max

                coppies = pltData.shape[0]
                spacing = X_max + X_max * loffset + 1

                ax = plt.subplot(111)
                plt.setp(ax, 'frame_on', False)
                ax.set_ylim([0, (Y_max-Y_min)+Y_max*koffset])
                ax.set_xlim([0, ((X_max - X_min)+X_max*loffset + 1)*coppies])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

                for c in np.arange(0,coppies):
                    for k in np.arange(Y_min, Y_max):    
                        for l in np.arange(X_min, X_max):
                            da = pltData[c][l][k]

                            
                            x_start = l + l*loffset + c*spacing 
                            x_stop = l+1 + l*loffset + c*spacing
                            y_start = k + k*koffset
                            y_stop = k+1 + k*koffset

                            
                            ax.imshow(da,cmap="binary",extent=[x_start,x_stop,y_start,y_stop],vmin=range_min, vmax=range_max)             

                plt.suptitle('Coupling coefficients for number {}'.format(i), fontsize=20)
                if save:
                    plt.savefig(self.destination + subdest + name + "_ID" +str(self.current_coupling) +"_"+ str(i) + ".png")
                    self.current_coupling += 1
                else:  
                    plt.show()
            
        

class CollectedData():
    """
    Possible types are:
        -image        
        -netwoek state
    """
    def __init__(self,type,data,epoch,batch):
        self.type = type
        self.data = data
        self.epoch = epoch
        self.batch = batch

    
    @staticmethod
    def find_first_occurance_index(target):
        found_index = {}
        target = target.numpy()

        for i in range(target.size):
            number = target[i]

            if number not in found_index:
                found_index[number] = i

        for i in range(10):
            if i not in found_index:
                found_index[i] = None

        return found_index 


    def __repr__(self):
            repr = "Type: {} | Epoch: {} | Batch: {} | Lables: {}".format(self.type,self.epoch,self.batch,self.lables)
            return repr
        

    

class InitConfig():
    def __init__(self):
        pass
    
    @staticmethod
    def check_config(f,args):
        try:
            run_type = str(args[1])
        except:
            print(
            "You have forgot to enter the type of execution parameter e.g. : \n",
            "python main.py exec_style \n"
            )
            sys.exit(1)            
        try:
            config = f["type"][run_type]
        except:   
            print(            
                "Configuration with this name does not exist in the config.jason file. \n",
                "You can create your own configuration by appending it to the list of existing ones \n",
                "Just make sure you follwo the pattern of other examples \n"            
                )
            sys.exit(1)

        return config