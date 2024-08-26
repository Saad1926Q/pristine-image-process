from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response,flash
import os
import cv2
import numpy as np
from models import Model
from flask_cors import CORS

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app=Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY']='3d6f45a5fc12445dbac2f59c3a6e7cb1'

model=Model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload',methods=["GET","POST"])
def upload():
    if request.method=="POST":
        if 'file' not in request.files:
            return jsonify({"error":"file doesnt exist","prediction":-1})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error":"Empty file","prediction":-1})
        if file and allowed_file(file.filename):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            model.image_to_feature_vector(image_get=image)
            pred=model.image_prediction(image=model.image)
            return jsonify({"prediction":pred})
        
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=4000)