#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:44:37 2019

@author: pratz
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.models import load_model

import cv2
import keras.backend as K
from PIL import Image, ImageFilter
import PIL

json_file = open('/home/pratz/Downloads/dataset/results/model28000.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/pratz/Downloads/dataset/results/model28000.h5")
print("Loaded model from disk")



"""Dir = '/home/pratz/Downloads/dataset/sketch-to-images-sketches/sketches/sketches'

for i,path in enumerate(os.listdir(Dir)):
    if i<10:
        im = cv2.imread(os.path.join(Dir,path),0)
        plt.imshow(im)
        plt.show()
        plt.imsave(path,im)
        im = cv2.resize(im,(64,64))
        im = im.reshape((1,4096))
        img = loaded_model.predict(im)[0]
        #img = cv2.medianBlur(img,3)
        #blur=cv2.GaussianBlur(img, (3,3), 0)
        #img = cv2.addWeighted(blur,1.5,img,-0.5,0)
        gen_img = (1/2.5) * img + 0.5
          
        cv2.imwrite("sz"+path,gen_img)
        plt.imsave("sz"+path,gen_img)
        plt.imshow(gen_img)
        plt.show()"""
       

def predict_photo(sketch_path):
    im = cv2.imread(sketch_path,0)
    plt.imshow(im)
    plt.show()    
    im = cv2.resize(im,(64,64))
    im = im.reshape((1,4096))
    img = loaded_model.predict(im)[0]
    #img=cv2.GaussianBlur(img, (3,3), 0)
    #img = cv2.addWeighted(blur,1.5,img,-0.5,0)
    gen_img = (1/2.5) * img + 0.5 
    #cv2.imwrite("/home/pratz",gen_img)
    plt.imsave("/home/pratz/save.jpg",gen_img)       
    plt.imshow(gen_img)
    plt.show()
    


    

