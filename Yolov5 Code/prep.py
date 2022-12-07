import pandas as pd
import os
import ast
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import cv2

import sys

DATASET_PATH = f'tensorflow-great-barrier-reef/yolo'

TRAIN_IMAGES_PATH = f"{DATASET_PATH}/images/train"
VAL_IMAGES_PATH = f"{DATASET_PATH}/images/val"

TRAIN_LABELS_PATH = f"{DATASET_PATH}/labels/train"
VAL_LABELS_PATH = f"{DATASET_PATH}/labels/val"


if not os.path.exists(TRAIN_IMAGES_PATH):
    os.makedirs(TRAIN_IMAGES_PATH)
if not os.path.exists(VAL_IMAGES_PATH):
    os.makedirs(VAL_IMAGES_PATH)
if not os.path.exists(TRAIN_LABELS_PATH):
    os.makedirs(TRAIN_LABELS_PATH)
if not os.path.exists(VAL_LABELS_PATH):
    os.makedirs(VAL_LABELS_PATH)

tqdm.pandas()
df = pd.read_csv("train-0.1.csv")


def num_boxes(annotations):
    annotations = ast.literal_eval(annotations)
    return len(annotations)

#df['num_bbox'] = df['annotations'].apply(lambda x: num_boxes(x))
#df = df[df.num_bbox > 0]

print("Starting")

def copy_file(row):
    if row.is_train:
        if not os.path.exists(f'{TRAIN_IMAGES_PATH}/{row.image_id}.jpg'):
            copyfile(row.image_path, f'{TRAIN_IMAGES_PATH}/{row.image_id}.jpg')
    else:
        if not os.path.exists(f'{VAL_IMAGES_PATH}/{row.image_id}.jpg'):
            copyfile(row.image_path, f'{VAL_IMAGES_PATH}/{row.image_id}.jpg')
            
_ = df.progress_apply(lambda row: copy_file(row), axis=1)

IMG_WIDTH, IMG_HEIGHT = 1280, 720

def get_yolo_format_bbox(img_w, img_h, box):
    w = box['width'] 
    h = box['height']
    
    if (bbox['x'] + bbox['width'] > 1280):
        w = 1280 - bbox['x'] 
    if (bbox['y'] + bbox['height'] > 720):
        h = 720 - bbox['y'] 
        
    xc = box['x'] + int(np.round(w/2))
    yc = box['y'] + int(np.round(h/2)) 

    return [xc/img_w, yc/img_h, w/img_w, h/img_h]
    

for index, row in tqdm(df.iterrows()):
    annotations = ast.literal_eval(row.annotations)
    bboxes = []
    for bbox in annotations:
        bbox = get_yolo_format_bbox(IMG_WIDTH, IMG_HEIGHT, bbox)
        bboxes.append(bbox)
        
    if row.is_train:
        file_name = f"{TRAIN_LABELS_PATH}/{row.image_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    else:
        file_name = f"{VAL_LABELS_PATH}/{row.image_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
    with open(file_name, 'w') as f:
        for i, bbox in enumerate(bboxes):
            label = 0
            bbox = [label]+bbox
            bbox = [str(i) for i in bbox]
            bbox = ' '.join(bbox)
            f.write(bbox)
            f.write('\n')
                

