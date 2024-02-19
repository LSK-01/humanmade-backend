# This file contains the functions for image preprocessing, feature extraction, etc.

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import cosine

def load_model():
    """ base_model = ResNet50(weights='imagenet', include_top=False)#exclude the final classification layer (top layer)
    model = Model(inputs=base_model.input, outputs=base_model.output) """
    # Load the pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    # Create a new model that will output the features from the 'conv1_relu' layer
    layer_name = 'conv4_block5_1_conv'  # Adjust this to the name of the layer you're interested in
    intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    return intermediate_layer_model



def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) #this is the target size of resnet
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(img_path, model):
    preprocessed_img = preprocess_image(img_path)
    print(preprocessed_img)
    features = model.predict(preprocessed_img)
    print(features)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

def calculate_similarity(feature_vec1, feature_vec2):
    return 1 - cosine(feature_vec1, feature_vec2)

def compare_images(img1_path, img2_path):
    model = load_model()
    features1 = extract_features(img1_path, model)
    features2 = extract_features(img2_path, model)
    return calculate_similarity(features1, features2)

""" print(compare_images('image1.png', 'image2.png'))
 """""" for i, layer in enumerate(model.layers):
    print(i, layer.name) """