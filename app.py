from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


INPUT_SHAPE=(128,128)
# MODEL_PATH ='models/vgg16_1.hdf5'
MODEL_PATH ='models/vgg16_1_bacillariophyceae.hdf5'
# MODEL_PATH ='models/effnet3_bacillariophyceae-nota.hdf5'

def load_my_model():
    global model

    # Load your trained model
    model = load_model(MODEL_PATH)

    # you have to compile model before use? 
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    # return loaded compiled model
    # return model


def prepare_image(image, target):
    # e.g target=(244,244)
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    input_arr = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    # return the processed image
    return input_arr


    print('Model loaded. View on http://127.0.0.1:5000/')
 


def model_predict(img_path, loaded_model):
    img = image.load_img(img_path, target_size=INPUT_SHAPE)
  
    # preprocess the image and prepare it for classification
    img = prepare_image(img, target=INPUT_SHAPE)

    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   
   #  x = preprocess_input(x, mode='caffe')

    preds = loaded_model.predict(img)
    return preds

# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'tester0' or request.form['password'] != 'meridion':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('index'))   # url_for('home')
    return render_template('login.html', error=error)

@app.route('/', methods=['GET'])
def default():
    # Main page
    return redirect(url_for('login'))
   # return render_template('index.html')

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


@app.route('/index', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# The API
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
         # load the model 
        load_my_model()
        preds = model_predict(file_path, model)

        class_labels = [
                            'Bacillariophyceae_Asterionella',
                            'Bacillariophyceae_Aulacoseira',
                            'Bacillariophyceae_Encyonema',
                            'Bacillariophyceae_Tabellaria']
        
                        #  [
                        #     'fragilariforma',
                        #     'frustulia',
                        #     'gyrosigma',
                        #     'melosira',
                        #     'meridion',
                        #     'parlibellus',
                        #     'petroneis',
                        #     'staurosirella',
                        #     'stephanodiscus'
                        # ]

        # Process your result for human
        pred_class = preds.argmax(axis=-1)  

        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result_index = str(pred_class[0])               # Convert to string
        
       
        print(pred_class)
        result  = class_labels[pred_class[0]]
        # return result
        return result 
    return None


if __name__ == '__main__':
    # Set debug to true if debugging
    app.run(debug=False)
