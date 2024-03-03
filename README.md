# C5 -  Multimodal recognition. Group 5

## Members

| Name                       | Email |
|----------------------------|-------|
| Anna Domènech Olivé        |  anna.domenecho@autonoma.cat|
| Noel Jiménez García        | noel.jimenez@autonoma.cat|
| Francisco A. Molina Bakhos | franciscoantonio.molina@autonoma.cat|
| Andrea Sánchez Sarrablo    | andrea.sanchezsar@autonoma.cat|


## Reports 

### Overleaf Project Report

- [Project Report](https://www.overleaf.com/read/dxygqtczmvrg#3e2723)

### Weekly Project Slides:

- [T1: Introduction to Pytorch - Image classification](https://docs.google.com/presentation/d/19ssEp37PrmSr4Sil_Iis9pQ4oS4kJtUTtwpkvF2iu-g/edit?usp=sharing)
- T2: Object Detection, recognition and segmentation
- T3: Image Retrieval
- T4: Cross-modal Retrieval
- T5: Diffusion models
- T6: Multimodal human analysis
- Final Presentation

### Code instructions

#### T2: Object Detection, recognition and segmentation - Usage

```bash
python main.py --action <action> --model <model>
```

Where:
```
<action> = ["simple_inference", "eval_inference", "random_search"]
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





