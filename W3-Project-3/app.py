from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
custom_model = load_model("/content/drive/MyDrive/Object_Detection/object_detection/cifar10_model.h5")

# CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to 32x32
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array.astype('float32') / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route to handle file upload and prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            try:
                # Save the file to disk
                filepath = os.path.join('/content/', file.filename)
                file.save(filepath)

                # Preprocess the image
                img_array = preprocess_image(filepath)

                # Make prediction
                predictions = custom_model.predict(img_array)
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_class_index]
                predicted_confidence = (predictions[0][predicted_class_index]) * 100

                # Remove the file after processing
                os.remove(filepath)

                return jsonify({
                    'predicted_class': predicted_class,
                    'predicted_confidence': f'{predicted_confidence:.2f}'
                })
            except Exception as e:
                return jsonify({'error': str(e)})
    return '''
    <!doctype html>
    <title>Upload an Image</title>
    <h1>Upload an image for CIFAR-10 classification</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# Run the app
if __name__ == '__main__':
   app.run(debug=False, port=8501)  # Explicitly specify the port

