# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:02:52 2018

@author: ( Jatin Bansal )
"""

from flask import Flask, render_template, Response
from ball_tracking import track_image

app = Flask(__name__)

base_path="/home/jatin/Downloads/Python/BTP/"




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/opencamera/',methods=['POST'])
def opencamera():
    track_image("")
    return ""


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port=5050,use_reloader=False,threaded=True)



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
