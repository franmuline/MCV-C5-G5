# T2: Object Detection, recognition and segmentation - Usage

### Setup

You can configure your `conda` environment from scratch by doing
```bash
conda create --name c5_w3 python=3.10
conda activate c5_w3
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
￼
conda install scikit-learn umap-learn faiss-gpu matplotlib
pip install opencv-python
```

### Code Instructions

```bash
python main.py --action <action> --model <model>
```

where:
```
<action> = ["simple_inference", "eval_inference", "random_search", "out_of_context"]
<model> = ["faster_rcnn", "mask_rcnn"]
```

For both `simple_inference` and `eval_inference` actions, **the user can specify
a path to the desired model weights in the file `Week 2/config_files/weights.yaml`**.
If it is not specified, the pre-trained weights provided by detectron2 will be used.

Simple inference: will run the selected model on the sequences of images of the KITTI-MOTS dataset provided in the `Week 2/config_files/chosen_sequences.yaml`. 
The sequences have to refer, by index, to the sequences of the validation set (indicated in `Week 2/config_files/dataset_config.yaml`).
Then, the results will be saved in the `Week 2/simple_inference/` folder, in a folder with the name of the model. 

Eval inference: will run the selected model on the validation set of the KITTI-MOTS dataset. 
The results of the evaluation will be saved in the `Week 2/output/` folder, in a folder with the name of the model, in a file called output.txt.

Random search: will run a random search to fine-tune the model changing the hyperparameters. 
It is conducted using wandb, and the hyperparameters are defined inside the function `random_search` in the file `Week 2/train.py`.

Out of context: will run Mask RCNN on the Out of context model (it has to be in the same directory as the whole project).





