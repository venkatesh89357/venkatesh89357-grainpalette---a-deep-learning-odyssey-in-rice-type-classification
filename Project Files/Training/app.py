import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from keras.layers import TFSMLayer

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load your model with TFSMLayer
model = TFSMLayer('rice_model', call_endpoint='serve')

df_labels = {0: 'arborio', 1: 'basmati', 2: 'ipsala', 3: 'jasmine', 4: 'karacadag'}

@app.route('/')
def home():
    return render_template('index.html')

# This route shows the upload form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files.get('image')
        if not f or f.filename == '':
            return render_template('predict.html', error="No file uploaded. Please select an image.")

        upload_folder = os.path.join(app.static_folder, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)

        # Preprocess image for model
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model(img)
        pred_class = np.argmax(prediction)
        pred_label = df_labels.get(pred_class, "Unknown")

        rel_path = os.path.relpath(filepath, start=os.path.abspath(os.path.join(__file__, '..', '..'))).replace("\\", "/")

        return render_template('details.html', prediction_text=pred_label, image_path='/' + rel_path)

    # GET request shows the form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
