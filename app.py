from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
import cv2
import base64
import numpy as np
from skimage.feature import graycomatrix
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

weights = {
    "histogram_colors": 0.2,
    "dominant_colors": 0.2,
    "color_moments": 0.2,
    "tamura_features": 0.2,
    "gabor_descriptors": 0.2
}

selected_image_data = {}
category_images_data = []

seuil = 0.2


def normalize_histogram(histogram):
    total = sum(histogram)
    if total == 0:
        return histogram  # Avoid division by zero
    return [float(bin_count) / total for bin_count in histogram]



def extract_tamura_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate Tamura features
    contrast = graycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
    directionality = graycomatrix(gray_image, [3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True,
                                  normed=True)
    coarseness = graycomatrix(gray_image, [5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    linelikeness = graycomatrix(gray_image, [7], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    regularity = graycomatrix(gray_image, [9], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    roughness = graycomatrix(gray_image, [11], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

    return {
        'contrast': np.mean(contrast),
        'directionality': np.mean(directionality),
        'coarseness': np.mean(coarseness),
        'linelikeness': np.mean(linelikeness),
        'regularity': np.mean(regularity),
        'roughness': np.mean(roughness)
    }


def extract_gabor_descriptors(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Gabor filter parameters
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    scales = [3, 6, 9]

    gabor_descriptors = []

    for scale in scales:
        for theta in orientations:
            # Apply Gabor filter
            gabor_filter = cv2.getGaborKernel((5, 5), scale, theta, 10.0, 1.0, 0, ktype=cv2.CV_32F)
            filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, gabor_filter)

            # Calculate mean and variance as descriptors
            mean_value = np.mean(filtered_image)
            variance_value = np.var(filtered_image)

            gabor_descriptors.extend([mean_value, variance_value])

    return gabor_descriptors


def calculate_moments(channel):
    moments = cv2.moments(channel)
    return {f'moment{key}': value for key, value in moments.items()}


@app.route('/api/process_image_data', methods=['POST'])
def process_image():
    data = request.get_json()
    print("Endpoint hit!")
    image_base64 = data.get('image_base64')
    if image_base64 is None:
        return jsonify({"error": "Image data missing"}), 400

    if not image_base64:
        return jsonify({"error": "Invalid image data"}), 400

    prefixes = {
        'data:image/jpg;base64,': 'jpg',
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
        try:
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                print("Error decoding image")
                return jsonify({"error": "Error decoding image"}), 400

            # Calculate histograms for each color channel (BGR)
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

            # Convert the image to RGB for color moments calculation
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process dominant colors
            pixels = image_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)

            # Calculate color moments for each channel
            color_moments = {
                "blue": calculate_moments(image[:, :, 0]),
                "green": calculate_moments(image[:, :, 1]),
                "red": calculate_moments(image[:, :, 2])
            }

            tamura_features = extract_tamura_features(image)
            gabor_descriptors = extract_gabor_descriptors(image)

            selected_image_data = {
                "histogram_colors": {
                    "blue": [int(value[0]) for value in hist_b],
                    "green": [int(value[0]) for value in hist_g],
                    "red": [int(value[0]) for value in hist_r]
                },
                "dominant_colors": {f"color{i}": color.tolist() for i, color in enumerate(dominant_colors)},
                "color_moments": list(color_moments.values()),
                "tamura_features": list(tamura_features.values()),
                "gabor_descriptors": {"values": gabor_descriptors}
            }

            return selected_image_data
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400


def normalize_feature(feature):
    if isinstance(feature, (int, float, np.integer, np.floating)):
        return np.array([feature])
    elif isinstance(feature, list) and all(
            isinstance(value, (int, float, np.integer, np.floating)) for value in feature):
        feature = np.array(feature).reshape(-1, 1)
        scaler = MinMaxScaler()
        return scaler.fit_transform(feature).flatten()
    elif isinstance(feature, str):
        # Handle non-numeric string feature
        return np.array([0.0])
    else:
        # Handle other non-numeric feature (like 'blue')
        return np.array([0.0])


def get_channel_distances(descriptor1, descriptor2, channel):
    def get_value(descriptor, channel):
        if isinstance(descriptor, dict):
            return descriptor.get(channel, [0, 0, 0])  # If it's a dictionary, get the value or default to [0, 0, 0]
        elif isinstance(descriptor, list):
            index = descriptor.index(channel) if channel in descriptor else -1
            return descriptor[index] if index != -1 else [0, 0,
                                                          0]  # If it's a list, get the value or default to [0, 0, 0]
        else:
            return descriptor


    def normalize_distance(distance, dimensions):
        return distance / np.sqrt(dimensions)

    value1 = get_value(descriptor1, channel)
    value2 = get_value(descriptor2, channel)

    # Handle case when values are strings
    if isinstance(value1, list) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in value1):
        value1 = normalize_feature(value1)
        value2 = normalize_feature(value2)

    # Handle case when values are strings
    if isinstance(value1, str) or isinstance(value2, str):
        return 0.0 if value1 == value2 else 1.0

    # Calculate and normalize Euclidean distance
    raw_distance = euclidean(value1, value2)
    normalized_distance = normalize_distance(raw_distance, len(value1))

    return normalized_distance


def calculate_descriptor_distance(descriptor1, descriptor2, descriptor_type):
    if not descriptor1 and not descriptor2:
        return 0.0
    elif not descriptor1 or not descriptor2:
        return 1.0

    def get_channels(descriptor, descriptor_type):
        if isinstance(descriptor, dict):
            # For dictionary-type descriptors, use keys as channels
            return descriptor.keys()
        elif isinstance(descriptor, list):
            # For list-type descriptors, use index positions as channels
            return range(len(descriptor))
        else:
            raise ValueError(f"Unsupported data structure for descriptor type: {descriptor_type}")

    channels1 = get_channels(descriptor1, descriptor_type)
    channels2 = get_channels(descriptor2, descriptor_type)

    if len(channels1) != len(channels2):
        raise ValueError("Descriptors do not have the same number of channels")

    distances = [get_channel_distances(descriptor1, descriptor2, channel) for channel in channels1]

    return np.mean(distances)


# The get_channel_distances function should be able to handle both dictionary and list inputs.


def calculate_feature_similarity(selected_feature, other_feature, descriptor_type):
    # Using 1 - distance as the similarity measure
    distance = calculate_descriptor_distance(selected_feature, other_feature, descriptor_type)
    similarity = 1 - distance


    print(similarity)

    return similarity


def calculate_similarity(selected_image_data, category_images_data, seuil,weights):
    similar_images = []

    for index, other_image_data in enumerate(category_images_data):
        feature_similarities = []

        for descriptor_type, weight in weights.items():
            selected_descriptor = selected_image_data.get(descriptor_type, {})
            other_descriptor = other_image_data.get(descriptor_type, {})

            if selected_descriptor and other_descriptor:
                feature_similarity = calculate_feature_similarity(selected_descriptor, other_descriptor,
                                                                  descriptor_type)
                feature_similarities.append(weights[descriptor_type] * feature_similarity)

        overall_similarity = sum(feature_similarities)

        # Checking if the overall similarity is above the threshold
        if overall_similarity >= seuil:
            similar_images.append({"index": index, "similarity": overall_similarity})

    return similar_images


@app.route('/api/calculate_similarity', methods=['POST'])
def calculate_similarity_route():
    data = request.get_json()
    selected_image_data = data.get('selected_image_data', {})
    category_images_data = data.get('category_images_data', [])
    weights = data.get('weights',{})
    seuil = data.get('seuil')
    if seuil is not None:
        try:
            seuil = float(seuil)
        except ValueError:
            # Handle the case where seuil is not a valid float
            print("Seuil is not a valid float")
    else:
        print("Seuil is not provided in the request")


    # Calculate similarity
    similar_images = calculate_similarity(selected_image_data, category_images_data, seuil,weights)

    # Return the result along with weights
    result = {
        "weights": weights,
        "similar_images": similar_images
    }

    return jsonify(result)


def calculate_value_association(similarity_values, user_feedback,i):
    is_good_feedback = user_feedback[i]["relevance"].lower() == "bon" if user_feedback else True
    sorted_indices = np.argsort(similarity_values)

    if is_good_feedback:
        value_association = {index: 1.2 - rank * 0.1 for rank, index in enumerate(sorted_indices)}
    else:
        value_association = {index: 1 for rank, index in enumerate(sorted_indices)}


    return value_association


def update_weights(old_weights, similarity_matrix, user_feedback):
    updated_weights = old_weights.copy()

    for feedback_entry in user_feedback:
        index = feedback_entry["index"]
        is_good_feedback = feedback_entry["relevance"].lower() == "bon"

        assigned_values = similarity_matrix[index]

        for feature in assigned_values:
            if feature in old_weights:

                updated_weights[feature] += old_weights[feature]

                if is_good_feedback:
                    updated_weights[feature] *= assigned_values[feature]
                else:
                    updated_weights[feature] *= assigned_values[feature]

    sum_weights = sum(updated_weights.values())
    normalized_weights = {key: value / sum_weights for key, value in updated_weights.items()}

    return normalized_weights


def calculate_similarity_matrix(selected_image_data, category_images_data, user_feedback):
    similarity_matrix = []

    i=0;

    for other_image_data in category_images_data:
        feature_similarities = {}

        for descriptor_type in selected_image_data.keys():
            selected_descriptor = selected_image_data.get(descriptor_type, {})
            other_descriptor = other_image_data.get(descriptor_type, {})

            if selected_descriptor and other_descriptor:
                feature_similarity = calculate_feature_similarity(selected_descriptor, other_descriptor,
                                                                  descriptor_type)
                feature_similarities[descriptor_type] = feature_similarity

        weight_association = calculate_value_association(list(feature_similarities.values()), user_feedback,i)
        i=i+1
        normalized_weights = {key: weight_association[index] for index, key in enumerate(feature_similarities)}

        similarity_matrix.append(normalized_weights)


    return similarity_matrix


def calculate_similarity_with_updated_weights(selected_image_data, category_images_data, updated_weights, seuil):
    similar_images = []

    for index, other_image_data in enumerate(category_images_data):
        feature_similarities = []

        for descriptor_type, weight in updated_weights.items():
            selected_descriptor = selected_image_data.get(descriptor_type, {})
            other_descriptor = other_image_data.get(descriptor_type, {})

            if selected_descriptor and other_descriptor:
                feature_similarity = calculate_feature_similarity(selected_descriptor, other_descriptor,
                                                                  descriptor_type)
                feature_similarities.append(weight * feature_similarity)

        overall_similarity = sum(feature_similarities)

        if overall_similarity >= seuil:
            similar_images.append({"index": index, "similarity": overall_similarity})

    return similar_images


@app.route('/api/update_weights_and_similarity', methods=['POST'])
def update_weights_and_similarity():


    data = request.get_json()

    selected_image_data = data.get('selected_image_data', {})
    category_images_data = data.get('category_images_data', [])
    user_feedback = data.get('user_feedback', [])

    print(selected_image_data)

    # Assuming the histogram is stored in a key 'histogram' in the data
    if 'histogram' in selected_image_data:
        selected_image_data['histogram'] = normalize_histogram(selected_image_data['histogram'])

    for image_data in category_images_data:
        if 'histogram' in image_data:
            image_data['histogram'] = normalize_histogram(image_data['histogram'])



    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(selected_image_data, category_images_data, user_feedback)
    # Update weights
    normalized_weights = update_weights(weights, similarity_matrix, user_feedback)

    # Calculate similarity with updated weights
    similar_images = calculate_similarity_with_updated_weights(selected_image_data, category_images_data,
                                                               normalized_weights, seuil)

    # Return the result
    return jsonify({"weights": normalized_weights, "similar_images": similar_images,"seuil":seuil})


if __name__ == '__main__':
    app.run(port=5000, debug=True)