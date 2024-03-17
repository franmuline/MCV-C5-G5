import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from datasets import load_dataset
from pathlib import Path

DEFAULT_PATH = './output/TripletNet/MIT_split/validation_features_and_labels.npy'
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

label_names = {
    0: 'Opencountry',
    1: 'Coast',
    2: 'Forest',
    3: 'Highway',
    4: 'Insidecity',
    5: 'Mountain',
    6: 'Street',
    7: 'Tallbuilding'
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
                        path:str=DEFAULT_PATH):
    # Visualize the UMAP embedding
    if nc == 2:
        plt.figure(figsize=(10, 8))
        for label, color in label_colors.items():
            indices = [i for i, c in enumerate(umap_colors) if c == color]
            if indices:
                plt.scatter(umap_embedding[indices, 0], umap_embedding[indices, 1], c=color, label=f'{label_names[label]}')
        plt.title('UMAP Visualization of ResNet Features (trained with Triplet)')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()  # Add legend with labels
    elif nc == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label, color in label_colors.items():
            indices = [i for i, c in enumerate(umap_colors) if c == color]
            if indices:
                ax.scatter(umap_embedding[indices, 0], umap_embedding[indices, 1], umap_embedding[indices, 2], c=color,
                           label=f'{label_names[label]}')
        ax.set_title('UMAP 3D Visualization of ResNet Features')
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        ax.legend()  # Add legend with labels

    # plt.show()
    plt.savefig(path + f'{nc}D_umap.png')


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
    umap = UMAP(n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbors, metric=metric)  # Reduce to 2 dimensions for visualization
    umap_embedding = umap.fit_transform(features)
    print(type(umap_embedding))
    path_to_save = str(Path(data_path).parent)
    if "train" in data_path:
        path_to_save += "/train_"
    else:
        path_to_save += "/validation_"
    plot_umap_embedding(umap_colors, umap_embedding, n_components, path_to_save)


def get_image(dataloader, idx):
    image = dataloader.dataset.__getitem__(idx)[0]
    label = dataloader.dataset.__getitem__(idx)[1]
    img = image.cpu().squeeze().numpy()
    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
    return img, label



def visualize_closest_images(data_path, indices_path, n=5):
    """Function to visualize the n closest images to a query image."""
    indices_np = np.load(indices_path)
    # Load the datasets without the shuffle
    train_data = load_dataset(data_path + "/train", 1, False)
    validation_data = load_dataset(data_path + "/test", 1, False)

    # Index chosen because ResNet50 (base model) had problems with this particular image
    # This way, we can see if it changes after metric learning
    random_rows = [102, 239, 458, 761]  # 102
    # We will leave a column of space between the query image and the closest images
    fig, axes = plt.subplots(len(random_rows), n + 2, figsize=(15, 2.7*(len(random_rows))),
                             gridspec_kw={'width_ratios': [1, 0.7, 1, 1, 1, 1, 1]})
    # Reduce space between rows
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # Iterate over the query images, we will plot a row for each query image, with the n closest images
    for i, random_row in enumerate(random_rows):
        # Get the query image and its label
        query_image, query_label = get_image(validation_data, random_row)
        # Plot the query image in the first column and the corresponding row
        axes[i, 0].imshow(query_image)
        if i == 0:
            axes[i, 0].set_title(f"Query image\n\n{label_names[query_label]}")
        else:
            axes[i, 0].set_title(f"{label_names[query_label]}")
        # Get the indices of the n closest images
        closest_images = indices_np[random_row, :n]
        # Plot the n closest images
        for j, idx in enumerate(closest_images):
            # Get the image and its label
            image, label = get_image(train_data, idx)
            # Plot the image
            axes[i, j + 2].imshow(image)
            if i == 0:
                axes[i, j + 2].set_title(f"Retrieved {j + 1}\n\n{label_names[label]}")
            else:
                axes[i, j + 2].set_title(f"{label_names[label]}")
    # For all axis, remove the axis
    for ax in axes.flatten():
        ax.axis("off")

    # plt.show()

    # Save on the same path where the indices are stored
    f_path = indices_path.split("/")[:-1]
    f_path = "/".join(f_path)
    # Reduce space between rows
    plt.savefig(f"{f_path}/closest_images.png", bbox_inches='tight')
