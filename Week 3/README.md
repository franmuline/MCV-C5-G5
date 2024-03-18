# T3: Image Retrieval

### Setup

You can configure your `conda` environment from scratch by doing
```bash
conda create --name c5_w3 python=3.10
conda activate c5_w3
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
￼
conda install scikit-learn umap-learn faiss-gpu matplotlib
pip install opencv-python
```

### Code Instructions

```bash
python main.py --action <action> [--ml_config <ml_cfg>]
```

where:
```
<action> = ["feature_extraction", "retrieval", "evaluation", "visualization", "closest_images", "metric_learning"]
<ml_cfg> = Any of the files in "config/metric_learning". Only for <action> = metric_learning.
```

#### Actions

`feature_extraction`: Perform the feature extraction of the images with the configuration defined in the `config/feature_extraction.yaml` file. The features will be stored in the directory defined in the yaml file.
- `model_path`: path of the model to use.
- `data`: name of the dataset located in the root of the workspace.
- `output_path`: output directory.

`retrieval`: Perform the image retrieval with the configuration defined in the `config/retrieval.yaml`. 
- `path_to_train_features` and `path_to_val_features`: path of the extracted features
- `method`: method employed to retrieve the most similar images from the database.
- `params`: 
    - `k`: number of nearest neighbors to retrieve.
    - `metric`: similarity / distance method to use.

`evaluation`: Evaluate the performance of the retrieved labels compared with the ground truth.
The configuration is set in the `config/evaluation.yaml` file.
- `path_to_features`: Path to the ground truth.
- `path_to_labels`: Path to the retrieved labels.

`visualization`: Visualize the feature vectors in 2D or 3D. The configuration is set in the `config/visualization.yaml` file.
- `type`: type of visualization. [`UMAP`]
- `path_to_features`: path to the features to visualize.
- `params`:
    - `n_components`:  number of dimensions to reduce to (2 or 3 for visualization).
    - `min_dist`:  minimum distance between points.
    - `n_neighbors`: number of nearest neighbours.
    - `metric`: distance to calculate the neigbours.

`closest_images`: Visualize what the closest images from the training set are for some of the query images. The configuration is set in the `config/closest_images.yaml` file.
- `data`: what dataset to use.
- `indices_path`: path to the retrieved indices. 

`metric_learning`: Perform the metric learning training with the configuration defined by `<ml_cfg>` (any of the files in `config/metric_learning`).
- `data`: dataset to use.
- `metric`: type of metric learning model (siames or triplet).
- `loss`: loss function to use (offline (Contrastive for Siamese and TripletLoss for Triplets) or online (OnlineContrastive for Siamese and OnlineTriplet for Triplets)).
- `loss_params`:
    - `margin`: margin for the contrastive loss.
    - `selector`: strategy to select the negative samples.
- `n_epochs`: number of epochs.
- `log_interval`: log interval.
- `batch_size`: batch size.