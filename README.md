# Road Segmentation 

## Description 

This repository contains the design and implementation of two convolutional neural networks to classify satellite images. More specifically, the goal is to separate 16x16 blocks of pixels between roads and the rest. 
The training set consists of 100 satellite images (400x400) with their respective ground truth. The testing set consists of 50 satellite images (608x608).
The first one is a traditional 4 layers CNN with increasing depth after each layer, the second one is an Unet architecture 

## Requirements

The code is tested with the following versions 
- TensorFlow 2.0
- Numpy 1.17.xx
- Pillow 6.2.xx
- Matplotlib 3.1.x
- Python 3.7.x

## Getting started


 ```bash 
git clone <repo_url> // clone the repo
cd src
python run.py -unet //for Unet architecture (To be used for course evaluation)
#python run.py -normal // for traditional architecutre
  ```


## Technical Details
  

|Characteristics|Unet|Standard|
|:---|---|---|
|Classification |Pixelwise classification|Patch classification|
|Input|Image RGB (H X W)|72 x 72 window centered surrounding the patch|
|Output|Image BW (H X W)|16 X 16 patch  |


## Training Hardware

The training was done using Google Colab with the following configuration
- GPU: 1 x NVIDIA Tesla P100 (16GB CoWoS HBM2 at 732 GB/s)
- CPU: 2 vCPU
- RAM: 12 Go

## File overview
* **run.py** : contains the steps to do to run our project and get a csv file submission in the end. In order to use this, type in the command line python3 run.py (-unet or -normal). 
* **unet.h5** : model trained with result F1 = 0.905
* **weights.h5** : weights of the CNN standard trained model with result F1 = 0.882 
* **final_submission.csv** : csv file generated through unet.h5.
* **helpers.py** : contains all the utilities functions used by the neural network.
* **train_xx.py** : contains the training code for the models 
## Authors
* Jalel zghonda
* Olivier Lam
* Ekin kulbray 

## References

Unet architecture : [Wikipedia](https://en.wikipedia.org/wiki/U-Net)