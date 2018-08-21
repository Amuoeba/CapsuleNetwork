import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from main import CUDA



class ImagePlotter():



    def __init__(self,destination="./images/",name="image_"):
        self.destination = destination
        self.current = 0
        self.name = name    

    def plot_images_separately(self,images,save=False):
        "Plot the six MNIST images separately."
        fig = plt.figure()
        
        num_images = images.shape[0]
        
        for j in range(num_images):
            ax = fig.add_subplot(1, num_images, j+1)
            ax.matshow(images[j], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        
        if save:
            fig.savefig(self.destination + self.name + str(self.current) + ".png")
            self.current += 1
        else:
            plt.show()