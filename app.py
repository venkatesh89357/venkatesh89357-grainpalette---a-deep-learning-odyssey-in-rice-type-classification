

import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import h5py
import numpy as np
import os
from flask import Flask, app, request, render_template
from tensorflow import keras
import cv2
import tensorflow_hub as hub

# Loading the saved model and initializing the flask app
model = tf.keras.models.load_model(
    filepath='rice.h5',
    custom_objects={'KerasLayer': hub.KerasLayer}
)
app = Flask(__name__)

# Render HTML pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files.get('image')
        if f is None or f.filename == '':
            return "No file uploaded"

        # File save path
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)

        # Image preprocessing
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)

        # Prediction
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        df_labels = {0: 'arborio', 1: 'basmati', 2: 'ipsala', 3: 'jasmine', 4: 'karacadag'}
        pred_label = df_labels.get(pred_class, "Unknown")

        return render_template('details.html', prediction_text=pred_label, image_path='/' + filepath)

    return render_template('details.html')



if __name__=="__main__":
    app.run(debug=True)  