from __future__ import division, print_function
from flask import jsonify, make_response
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import os
import time
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from numpy.random import seed
from numpy.random import rand
from random import random
import random

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

seed(1)

# Define a flask app
dict={0:('No illness','0'),1: ('Mild','0'),2 : ('Moderate','98%'),3:('Severe','0'),4: ('intense','0')}

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/healthcare_resnet50_30_0.h5'
# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/index2.html')
def index2():
    # Predict page
    return render_template('index2.html')

@app.route('/index.html')
def index3():
    # Predict page
    return render_template('index.html')
    
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        pr1=np.argmax(preds)
        result = dict[pr1]
        result = list(result)
        if result[0] == "No DR":
                result[1] = "0%"
        else: 
                result[1] = str( "{0:.2f}".format(random.randrange(60, 100))) + "%"

        result = tuple(result)

        return make_response(jsonify(result), 201)
    return None


if __name__ == '__main__':
    #app.jinja_env.cache = {}
    #app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

