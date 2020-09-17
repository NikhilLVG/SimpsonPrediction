#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:32:05 2020

@author: nikki
"""

import os 
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

classes = r'/home/nikki/Desktop/python/ML/simpsons/validation/'
my_dirs = [d for d in os.listdir(classes) if os.path.isdir(os.path.join(classes,d))]

my_dirs
img = image.load_img(r"/home/nikki/Desktop/python/ML/simpsons/validation/ned_flanders/ned_flanders_28.jpg",target_size=(32,32))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model(r'/home/nikki/Desktop/python/ML/simpsons_vgg16.h5')
#output = saved_model.predict_classes(img)[0]
output = classider.predict_classes(img)
my_dirs[output[0]]


plt.title(my_dirs[output])
plt.show()