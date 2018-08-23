import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import itertools


plt.switch_backend('agg')

# from main import CUDA


class PrepareExperiment():
    def __init__(self, id, home="."):
        self.id = id
        self.home=home        
        self.expHome = None
        self.images = None
        self.plots = None
        self._prepare_foldiers()
    
    def _prepare_foldiers(self):
        dirname = self.unique_name(self.home + "/experiments/experiment",makedir=True)
        images = dirname+"/images"
        plots = dirname+"/plots"       
        os.makedirs(dirname)        
        os.makedirs(images)
        os.makedirs(plots)
        self.expHome = dirname+"/"
        self.images = images+"/"
        self.plots = plots+"/"
        
        

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


class ImagePlotter():

    def __init__(self,destination="./images/",name="image_"):
        self.destination = destination
        self.current = 0
        self.name = name    

    def plot_images_separately(self,images,save=False,name="default"):
        "Plot the six MNIST images separately."
        fig = plt.figure()
        
        num_images = images.shape[0]
        
        for j in range(num_images):
            ax = fig.add_subplot(1, num_images, j+1)
            ax.matshow(images[j], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        
        if save:
            fig.savefig(self.destination + name + str(self.current) + ".png")
            self.current += 1
        else:
            plt.show()




