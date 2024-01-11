import pickle
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import pandas as pd
import os
import models as model_list
import tasks
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import offsetbox
from sklearn import manifold
from PIL import Image


def plot_central_frames(X, Z):
    
    '''
    X: reduced_features: n_records*1 x 1 x 2
    Z: central_frames_images: central frames for each record  
    '''

    print(X.shape[0])
    x_coords = X[:, 0]
    y_coords = X[:, 1]

    plt.figure()
    ax = plt.subplot()
    ax.scatter(x_coords,y_coords) 
    
    for x , y, z in zip(x_coords, y_coords, Z):
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(z, zoom=0.07),(x, y))
        ax.add_artist(imagebox)
       
    plt.xticks([]), plt.yticks([])  
    # Add labels and title to the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('frames in 2D Space')
 
    plt.show()

 
def plot_actions(X, Y):
    
    '''
    X: reduced_features: n_records*1 x 1 x 2
    Y: useful_dataset: uid*args_num_clips (columns: uid, central_frame, verb)
    '''
    # Extract the x and y coordinates from the vector
    x_coords = X[:, 0]
    y_coords = X[:, 1]

    #Assign a color to each unique verb
    unique_verbs = Y.verb.unique()
    
    # retrieve a categorical colormap ('tab40c') that automatically assigns colors to categories. This one has 40 different colors (try also 'tab40c')
    cmap = get_cmap(lut = 24)
    
    for i, verb in enumerate(unique_verbs):
        mask = (Y.verb == verb)
        plt.scatter(x_coords[mask], y_coords[mask], c=cmap(i), label=verb)
        
    # Add labels and title to the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Verbs in 2D Space')

    #legend
    plt.legend()
    # Show the plot
    plt.show()  

def main(): 
    """
    I3D returns features of dimesnion: n_records (1345) x num_clips x 1 x 1014
    This function reads saved features dimesion, applies dimensionality reduction (t-SNE) and plots features in 2D space
    """                 
    #Use I3D_save_features.yaml                                                                        
    full_saved_features = EpicKitchensDataset(args.dataset.shift.split("-")[1], 
                                              args.modality,
                                              args.split, 
                                              args.dataset,
                                              args.save.num_frames_per_clip,
                                              args.save.num_clips, 
                                              args.save.dense_sampling,
                                              transform=None, 
                                              load_feat=True, #!!
                                              additional_info=True, 
                                              **{"save": args.split})
    
    saved_features = np.array(full_saved_features.model_features["features_RGB"].tolist()) 
    #average the num_clips features for each record
    saved_features = np.mean(saved_features, axis=1)
    saved_features.reshape(-1*1024)
    
    #Dimensionality reduction
    #*PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(saved_features)
    
    #*tSNE
    #tsne = manifold.TSNE(n_components=2, random_state=0)
    #reduced_features = tsne.fit_transform(saved_features)
       
    #extract central frame for each record
    central_frames = (full_saved_features.model_features.stop_frame - full_saved_features.model_features.start_frame) // 2
    
    #dataset for the plotting 
    useful_dataset = full_saved_features.model_features.loc[:, ["uid", "verb"]]
    useful_dataset["central_frames"] = central_frames
    #?print(useful_dataset.shape)
    
    #load each central frame image for each record
    central_frames_images = list()
    for modality in full_saved_features.modalities:
        for i, record in enumerate(full_saved_features.video_list):
            central_frames_image = full_saved_features._load_data(modality, record, central_frames[i] )
            central_frames_images.extend(central_frames_image)
        
    #plot
    plot_actions(reduced_features, useful_dataset)
    plot_central_frames(reduced_features, central_frames_images)     
   

    return 0


if __name__ == '__main__':
    main()
