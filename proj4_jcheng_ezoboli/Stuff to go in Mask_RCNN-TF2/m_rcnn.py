#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 21:14:56 2022

@author: justincheng
"""

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 81

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
rcnn.load_weights('model.h5', by_name=True)

images = ["enrico_cat", "room", "room3"]

for i in images:

    img = img_to_array(load_img(i+'.jpg'))
    
    results = rcnn.detect([img], verbose=0)
    
    data = pyplot.imread(i+'.jpg')
    pyplot.imshow(data)
    ax = pyplot.gca()
    
    for box in results[0]['rois']:
         y1, x1, y2, x2 = box
         width, height = x2 - x1, y2 - y1
         rect = Rectangle((x1, y1), width, height, fill=False, color='red')
         ax.add_patch(rect)
    pyplot.savefig(i  + '.png')