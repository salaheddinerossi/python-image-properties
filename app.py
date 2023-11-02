from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.cluster import KMeans
import requests
import cv2
import json
import base64
import numpy as np

  

app = Flask(__name__)
CORS(app)

# The base URL of the Express.js backend


@app.route('/api/process_image_data', methods=['POST'])
def process_image_data():
    combined_data = {
        "dominant_colors": {},
        "color_moments": {},
        "histogram_colors": {
            "blue": {},
            "green": {},
            "red": {}
        }
    }

    data = request.get_json()
    print("Endpoint hit!")

    image_base64 = data.get('image_base64')

    prefixes = {
        'data:image/jpeg;base64,': 'jpeg',
        'data:image/png;base64,': 'png',
        'data:image/gif;base64,': 'gif',
        'data:image/bmp;base64,': 'bmp',
        'data:image/webp;base64,': 'webp'

    }

    for prefix in prefixes:
        if image_base64.startswith(prefix):
            image_base64 = image_base64.replace(prefix, '')
            break

    if image_base64:
        # Decode the base64 string to get the image data

        image_data = base64.b64decode(image_base64)

        # Convert image data to a numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Calculate histograms for each color channel (BGR)
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Store histogram data in the combined_data dictionary
    combined_data["histogram_colors"]["blue"] = [int(value[0]) for value in hist_b]
    combined_data["histogram_colors"]["green"] = [int(value[0]) for value in hist_g]
    combined_data["histogram_colors"]["red"] = [int(value[0]) for value in hist_r]

    # Convert the image to RGB and process dominant colors
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Store dominant colors data in the combined_data dictionary
    combined_data["dominant_colors"] = {
        f"color{i}": color.tolist() for i, color in enumerate(dominant_colors)
    }

    # Calculate color moments for each channel and store the data
    color_moments = []
    for channel in range(3):
        channel_image = image[:, :, channel]
        moments = cv2.moments(channel_image)
        color_moments.append(moments)

    combined_data["color_moments"] = [
        {f'moment{key}': value for key, value in moments.items()}
        for moments in color_moments
    ]

    # Return a JSON response upon processing image data
    return jsonify(combined_data)



if __name__ == '__main__':
    app.run(port=5000,debug=True)
