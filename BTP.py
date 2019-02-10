#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 23:37:48 2019

@author: jatin
"""


from ball_tracking import track_image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt


classes_size=64
image_size=224
path='/home/jatin/Downloads/Python/BTP/'
save_location="/home/jatin/Downloads/Python/BTP/result.png"
model_path=path+'weights'+str(classes_size)+'.best.hdf5'



def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()
    return img_tensor


def predict_image(img_path,model_path):
    model = load_model(model_path)
    labels = np.load(path+'labels.npy').item()
    new_image=load_image(img_path)
    pred = model.predict(new_image)
    answer = np.argmax(pred)
    result=labels[answer]
    
    return result


if_image_exist=track_image(save_location)

if if_image_exist:
    prediction=predict_image(save_location,model_path)
    print(prediction)





