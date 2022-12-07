First, we need to install required libraries:

torch
opencv-python
pandas
numpy
matplotlib
albumentations
collections

This is in a form of scripts instead of ipynb because ASTI requires codes to be converted as scripts so that we can run it using a command.

Although CSV files have been prepared and created, CSVPrepare.py is the one that preprocess CSV files containing annotations of the training images into its appropriate from for faster image reading.

train.py is the one that trains the Faster-RCNN. it is just one-click-run

Dataset is not available in this directory due to its very large size. However, it is still available on this link (Until 2021 Feb 18).

Please download it and add it to \tensorflow-great-barrier-reef\train_images
https://www.kaggle.com/c/tensorflow-great-barrier-reef/data

The model is created in ..\Faster RCNN\tensorflow-great-barrier-reef\rcnn_models_100 named
fasterrcnn_resnet50_fpn-e99.bin
use this in inference.ipynb
inference.ipynb validates the validation images defined in train.csv. It tests the last model created from train.py