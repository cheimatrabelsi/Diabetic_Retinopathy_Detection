from flask import Flask, render_template, request, jsonify
import sys
import tensorflow as tf
import PIL.Image
import numpy as np

app = Flask(__name__, static_url_path="/static")

model = tf.keras.models.load_model('your_model.h5')


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    print("Received a POST request"),

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        image_file = request.files['image']
        image = PIL.Image.open(image_file)
        image = image.resize((256, 256))

        # Convert image to array, normalize, and reshape
        image_array = np.asarray(image, dtype=np.float32) / 255
        image_array = image_array.reshape(1, 256, 256, 3)  # Reshape to a single-sample batch

        # Assigning label names to the corresponding indexes
        labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3: 'Proliferate_DR', 4: 'Severe'}
        # Make predictions using the model
        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions)
        predicted_class_idx_scalar = predicted_class_idx.item()  # Extract the scalar value
        predicted_class = labels[predicted_class_idx_scalar]
        print(predicted_class)
        return jsonify({'prediction': predicted_class })
    except Exception as e:
        return jsonify({'error': str(e)})
    # return(render_template('index.html'))


if __name__ == '__main__':
    app.run()
