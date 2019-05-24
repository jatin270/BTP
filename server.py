# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:02:52 2018

@author: ( Jatin Bansal )
"""

from flask import Flask, render_template
from ball_tracking import track_image
from image_prediction import predict_image,predict_caption

app = Flask(__name__)

save_location="/home/jatin/Downloads/Python/BTP/BTP/result.png"
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
    
if __name__ == "__main__":
    app.debug = False
    app.run(host='0.0.0.0',port=5058,threaded=False)


