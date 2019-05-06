# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:02:52 2018

@author: ( Jatin Bansal )
"""

from flask import Flask, render_template, Response
from ball_tracking import track_image

app = Flask(__name__)

base_path="/home/jatin/Downloads/Python/BTP/"
save_location="/home/jatin/Downloads/Python/BTP/BTP/result.png"




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/opencamera/',methods=['POST'])
def opencamera():
    print("Hello")
    track_image(save_location)
#    label=predict_image(save_location)
    return "apple"

@app.route('/getcaption/',methods=['GET'])
def getcaption():
    return "Hi caption here"

#
#def predict_image(save_location):
#    labels = np.load(label_path).item()
#    model = load_model(model_path)
#    new_image=load_image(save_location)
#    pred = model.predict(new_image)
#    answer = np.argmax(pred)
#    result=labels[answer]
#    return result
#            

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port=5057,threaded=True)


#
#
#@app.route("/train_dataset/<size>")
#def train_dataset(size):
#    return "Successfully trained"
#
#
#@app.route("/getstats/")
#def get_stats():
#    return ""
#
#@app.route("/test_dataset",methods=['POST'])
#def test_dataset():
#    data = request.values.to_dict(flat=False)
#    print(data['question'][0])
#    return ""
#
