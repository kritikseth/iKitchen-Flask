import os
import cv2 
import keras
import tensorflow as tf
import efficientnet.keras as efn
import numpy as np 
from flask import Flask, flash, jsonify, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
import mapping as mp
import predict as pred

# Configure application
app = Flask(__name__)
UPLOAD_FOLDER = 'static'

# Ensure templates are auto-reloaded
app.config['TEMPLATES_AUTO_RELOAD'] = True

model = keras.models.load_model('static/models/effnet.hdf5')

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = 0
    response.headers['Pragma'] = 'no-cache'
    return response

# Configure session to use filesystem (instead of signed cookies)
app.config['SESSION_FILE_DIR'] = mkdtemp()
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


@app.route('/', methods=['GET', 'POST'])
def img_predict():
    '''Run model and show outcome'''
    if request.method == 'POST':
        image_file = request.files['img']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)

            image = pred.preprocess(image_location)
            result = pred.predict(model, image)

            return render_template('index.html', image_loc=image_file.filename, res=result)

    return render_template('index.html', image_loc=None, res=None)



if __name__ == '__main__':
    app.run(debug=True)
