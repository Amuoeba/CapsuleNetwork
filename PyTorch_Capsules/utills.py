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
    """
    A class used to prepare the experiment layout. It allso contains methods to plot graphs and images
    for an experiment.
    Arguments:
        id: Just some identifier
        home: Home foldier for your experiment results
    Output:
        A foldier structure rooted at the specified home address
    
    """

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
        
    
    def flush_collected_data(self):
        """
        Clears the data from the collected data bufer
        """
        self.additional_collected_data = []

    def _prepare_foldiers(self):
        """
        This method prepares the foldier structure for the experiments
        """
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

        for i in range(10):
            os.makedirs(img+self.coupling_image_dest+str(i))

    @staticmethod
    def unique_name(basename, ext=None, makedir=False):
        """
        This method is used to define unique foldier names if some name is already in that directory
        Args:
            basename: the base name of the file/foldier
            ext: file extention if you are creating files
            makedir: 
                If set to True you are implying that you are creating directories, else you 
                are createing files and the result my varry
        output:
            The unique name for the file/directory based on the basename
        """
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
    
    def create_plots(self,verbose=False):
        """
        A class instance method that creates all the plots for the experiment. Data for the plots is read
        from the class instance fields.
        Args:
            None
        Output:
            Experiment images in appropriate foldiers
        """

        counter = 1
        len_data = len(self.additional_collected_data)

        for data in self.additional_collected_data:
            image = data["image"]
            coupling = data["coupling"]

            if verbose:
                print("Ploting data: {}/{}".format(counter,len_data,len(image.data),len(coupling.data)),end="\n",flush=False)         

            reconst_image_name = "image_" + str(image.epoch) + "_"
            self.plotter.plot_reconstruction_images(image,save=True,name=reconst_image_name,subdest=self.reconst_image_dest,verbose=verbose)
            couple_image_name = "coupl_" + str(coupling.epoch) + "_BA" + str(coupling.batch)
            self.plotter.plot_coupling_image(coupling,save=True,name=couple_image_name,subdest=self.coupling_image_dest,verbose=verbose)

            counter += 1

            

        
        
        self.plot_train_data()



    def plot_train_data(self):
        """
        A class instance method that plots just the graphs of all final collected datapoints
        TODO: Combine with image plotter so there is only one class for plots in the future
        """
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
    """
    A class that contains all the plotting functions for images. Its purpuse is to have all functions in one 
    class and that it produces unique plot names based on how many times certain plotting methods were called
    from a certain instantiation of the ImagePlotter()
    Args:
        destination: The directory where images should be saved
        name: Prefered default name for images
    """
    def __init__(self,destination="./images/",name="image_"):
        self.destination = destination
        self.current_reconst = 0
        self.current_coupling = 0
        self.name = name    

    def plot_reconstruction_images(self,images,save=False,name="default",subdest="",verbose=False):
        """
        Class instance method that plots the reconstructed images.
        Args:
            images:
                Data for the images as a CollectedData class. For a single image the data within a class
                is a (1,28,28) dimensional numpy array
            save: If set to True the images will be saved instead of displayed
            name: Prefered default name for the images
            subdest: Subdirectory for the images
        Output:
            A figure of 10 reconstructed digits ordered from 0 to 9. If the image is absent from the
            data a blank white image is presented in its place.
        """    
        
        images = images.data
        fig = plt.figure()
        len_data = len(images)

        for j in range(10):

            if verbose:
                print("Processing reconstruction example {}/{}".format(j+1,len_data),end="\r"),

            if not images[j] is None:
                img_data =np.squeeze(images[j])
            else:
                img_data = np.zeros((28,28))
            
            ax = fig.add_subplot(1, 10, j+1)
            ax.matshow(img_data, cmap = matplotlib.cm.binary)
            ax.set_title("#{}".format(j))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        
        if verbose:
            print()

        if save:
            # print(self.destination)
            # print(subdest)
            # print(name)
            fig.savefig(self.destination + subdest + name + str(self.current_reconst) + ".png")
            self.current_reconst += 1
            fig.clf()
            plt.clf()
        else:
            plt.show()
    
    def plot_coupling_image(self,data,save=False,name="default",subdest="",verbose=False):
        """
        A plotting function that represents the coupling coefficents in between two adjecent capsule
        layser. The function plots a (10,32,6,6) image for each itteration. The strenght of the coupling
        is represented as the darkness of the spots in the image. The darker the spot the stronger is
        that capsule coupled with the capsule in the next layer (each column represents a capsule in the 
        next layer)
        Args:
            data: 
                As a CollectedData class. Each datapoint is a numpy array of sze 
                (num_itterations,num_next_layer, num_prev_layer,6,6)
            save: If set to True the plots are saved instead od displayed
            name: Preffered default name
            subdest: Subdirectory for saving the images
        """
        all_data = data.data
        len_data = len(all_data)


        for i in range(10):
            if verbose:
                print("Processing coupling example: {}/{}".format(i+1,len_data),end="\r")

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

                            
                            im = ax.imshow(da,cmap="binary",extent=[x_start,x_stop,y_start,y_stop],vmin=range_min, vmax=range_max)             

                plt.suptitle('Coupling coefficients for number {}'.format(i), fontsize=50)
                plt.colorbar(im)

                if save:
                    plt.savefig(self.destination + subdest + str(i) + "/" + name + "_ID" +str(self.current_coupling) +"_"+ str(i) + ".png")
                    self.current_coupling += 1
                    plt.clf()
                else:  
                    plt.show()
        
        if verbose:
            print()
            
        

class CollectedData():
    """
    A helper class for data collection from that encapsualtes all the data that was collected
    from a network at a certain collection point.
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
        """
        A method to extract the indices of first occurance of a digit in a batch.
        If the number is not found its index is None:
        Args:
            target: A tensor of target values
        Out:
            A dictionary of type {number:index} that contains all the numbers
            from 0 to 9
        """
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
    """
    Contains methods for checking the console input of the user
    """
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