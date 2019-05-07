# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:02:52 2018

@author: ( Jatin Bansal )
"""

from flask import Flask, render_template
from ball_tracking import track_image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from numpy import argmax
from pickle import load



app = Flask(__name__)

classes_size=64
image_size=224

base_path="/home/jatin/Downloads/Python/BTP/"
save_location="/home/jatin/Downloads/Python/BTP/BTP/result.png"
model_file_path="/home/jatin/Downloads/Python/BTP/BTP/model/"
caption_photo_path="/home/jatin/Downloads/sample.jpg"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/opencamera/',methods=['POST'])
def opencamera():
    print("Hello")
    track_image(save_location)
    label=predict_image(save_location)
    print(label)
    return label

@app.route('/getcaption/',methods=['POST'])
def getcaption():
    caption=predict_caption(caption_photo_path)
    print(caption)
    return caption


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

def predict_image(save_location):
    labels = np.load(model_file_path+"labels.npy").item()
    model = load_model(model_file_path+'weights'+str(classes_size)+'.best.hdf5')
    new_image=load_image(save_location)
    pred = model.predict(new_image)
    answer = np.argmax(pred)
    result=labels[answer]
    return result


def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, photo, max_lengt):
	in_text = 'startseq'
	for i in range(max_lengt):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_lengt)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text


def predict_caption(photo_path):
    tokenizer = load(open(model_file_path+"tokenizer.pkl", 'rb'))
    max_length = 34
    model = load_model(model_file_path+"model.h5")
    photo = extract_features(photo_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    return description
    
if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port=5058,threaded=True)


