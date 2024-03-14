import os
import torch
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

DEFAULT_PATH = 'MIT_split_train_features_and_labels.npy'
label_colors = {
    0: '#ff0000',  # Red
    1: '#00ff00',  # Green
    2: '#0000ff',  # Blue
    3: '#ffff00',  # Yellow
    4: '#ff00ff',  # Magenta
    5: '#00ffff',  # Cyan
    6: '#800080',  # Purple
    7: '#008000'   # Dark Green
}

def read_features_labels(path: str = DEFAULT_PATH):
    '''
    input: path to .npy file with features and labels
    output: features, labels, colors to plot the labels
    
    '''
    data = np.load(path)
    labels = data[:,-1:]
    features = data[:,:-1]

    return features, labels

def generate_color_list(labels: list, available_colors: dict=label_colors):
    return [available_colors[int(label[0])] for label in labels]

def plot_umap_embedding(umap_colors: list,
                        umap_embedding: np.ndarray,
                        nc:int=2,
                        ):
    # Visualize the UMAP embedding
    if nc == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=umap_colors)
        plt.title('UMAP Visualization of ResNet Features')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
    elif nc == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for 3D visualization
        ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], umap_embedding[:, 2], c=umap_colors)

        ax.set_title('UMAP 3D Visualization of ResNet Features')
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')

    plt.show()


def visualize_UMAP(data_path:str=DEFAULT_PATH,
                   n_components:int=2,
                   min_dist:float=0.1,
                   n_neighbors:int=50,
                   metric:str='manhattan'):
    # Read file with stored features and labels
    features, labels = read_features_labels(data_path)
    # Generate list of colors for the plot
    umap_colors = generate_color_list(labels)

    #Dimensionality reduction with UMAP
    umap = UMAP(n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbors, metric='manhattan')  # Reduce to 2 dimensions for visualization
    umap_embedding = umap.fit_transform(features)
    print(type(umap_embedding))
    plot_umap_embedding(umap_colors,umap_embedding,n_components)
    
# Si n_components es 2 es 2D i si es 3 es 3D
visualize_UMAP(n_components=3,min_dist=0.1,n_neighbors=50)




