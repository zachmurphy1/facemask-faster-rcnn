# Facemask Faster R-CNN

John Morkos, Andy Ding, and Zach Murphy

This repo contains code that implements a Faster R-CNN network to detect facemasks in images and classify into 3 classes:
1. No mask
2. Masked
3. Mask worn, but incorrectly.

## Setup -- you must do these steps to run this code
The notebooks in this repo are linked to data in a Google Drive directory. This directory contains raw and preprocessed code, saved models, and results. To access this directory and run this code, you __MUST__:
1. Request access to this folder at the following link: https://drive.google.com/drive/folders/1Hu9UOCsRE-5A_7U7IRsCEcNeqM416GmW?usp=sharing
2. From your "Shared with me" section of Google Drive, right click on the `facemask-faster-rcnn` directory and click `Add shortcut to Drive`.

## Overview
The notebooks should be run the following order:
1. Train_Val_Test_Split.ipynb
  - Performs 60-20-20% train-val-test split.
2. Oversampling.ipynb
  - Performs oversampling on the train dataset. Underrepresented instances were cropped with a border 10 pixels greater than the ground truth bounding box in all directions. Crops were randomly selected and inserted into a random image with replacement. To handle different image scales, crops were scaled to the mean width of the ground truth boxes in the target image times a random factor between 0.8 and 1.2. To prevent large discrepancies in resolution, scale factors were constrained to be between 0.25 and 4. Scaled crops were randomly placed into images such that the cropped image plus an extra boundary of 1/20th the target image width had an IOU of 0 with all ground truth boxes already in the image. The gaps between each class and the maximal class (masked) were closed by 3/4 (since the above placement and scale restraints did not permit closing the gaps entirely).
3. Super_Resolution.ipynb
  - Trains a SRResNet super-resolution network for 4x upscaling using 5k 128x128 px images of faces from the Flickr Faces HQ (FFHQ) dataset and validates on 1,000 separate images from FFHQ.
4. Faster_R_CNN.ipynb
  - Trains main Faster R-CNN object detectiion model
5. Evaluation.ipynb
  - Implements mean average precision (mAP) and average precision (AP) by class using the PASCAL VOC 2010-2012 AUC method. Plots loss curves and precision-recall curves.

## Four different models
We optimized four different models:
- Bare-bones (BB): Only Faster R-CNN with color jitter and random horizontal flips on the train set
- Super resolution (SR): BB + preprocessing all inputs to upscale by 4x
- Oversampling (OS): BB + oversampled train set to balance classes
- Super resolution + Oversampling (SR+OS): Combines super resolution preprocessing and train set oversampling on top of BB

To implement super resolution, set `sr=True` in the data set instantiation within Faster_R_CNN.ipynb.

To implement oversampling, set `op=train/oversampling` in the train set instantiation within Faster_R_CNN.ipynb.
