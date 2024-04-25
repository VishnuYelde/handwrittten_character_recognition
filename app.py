from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model('./model/best_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (28, 28))
    normalized_image = resized_image / 255.0
    input_image = normalized_image.reshape(1, 28, 28, 1)
    prediction = model.predict(input_image)
    predicted_label = chr(65 + np.argmax(prediction))  # Assuming labels are from A to Z
    return render_template('index.html', result=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)