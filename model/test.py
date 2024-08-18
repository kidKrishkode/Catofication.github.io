import sys
import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import Preprocessor
import ast

# # Load images and labels from the weights folder
# def load_images_and_labels(weights_folder):
#     images = []
#     labels = []
    
#     for folder_name in os.listdir(weights_folder):
#         folder_path = os.path.join(weights_folder, folder_name)
#         meta_file = os.path.join(folder_path, "meta.json")
        
#         if os.path.exists(meta_file):
#             with open(meta_file, 'r') as file:
#                 data = json.load(file)
                
#                 for entry in data:
#                     image_path = os.path.join(folder_path, entry['path'])
#                     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
#                     if image is not None:
#                         image = cv2.resize(image, (128, 128))  # Resize the image
#                         image = image.flatten()  # Flatten the image (128x128 -> 16384)
#                         images.append(image)
#                         labels.append(folder_name)  # Label based on folder (cat/dog)
    
#     return np.array(images), np.array(labels)

# # Preprocess images and split into train/test sets
# def preprocess_data(images, labels):
#     # Convert categorical labels to binary (0: cat, 1: dog)
#     labels = np.array([1 if label == "dogs" else 0 for label in labels])
    
#     # Split data into train and test sets
#     return train_test_split(images, labels, test_size=0.5, random_state=20)

# # Train the model using Logistic Regression
# def train_model(X_train, y_train):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     return model

# # Predict the class of the image
# def classify_image(image_path, model):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (128, 128))
#     image = image.flatten()  # Flatten the image for prediction
    
#     prediction = model.predict([image])[0]
    
#     if prediction == 1:
#         return "Dog"
#     else:
#         return "Cat"


# Load images and labels from the weights folder
def load_images_and_labels(weights_folder):
    images = []
    labels = []
    
    for folder_name in os.listdir(weights_folder):
        folder_path = os.path.join(weights_folder, folder_name)
        meta_file = os.path.join(folder_path, "meta.json")
        
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as file:
                data = json.load(file)
                
                for entry in data:
                    image_path = os.path.join(folder_path, entry['path'])
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None:
                        image = cv2.resize(image, (128, 128))  # Resize the image
                        image = image.flatten()  # Flatten the image (128x128 -> 16384)
                        images.append(image)
                        labels.append(folder_name)  # Label based on folder (cat/dog)
    
    return np.array(images), np.array(labels)

# Preprocess images and split into train/test sets
def preprocess_data(images, labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Split data into train and test sets
    return train_test_split(images, labels, test_size=0.5, random_state=15), label_encoder

# Train the model using Logistic Regression
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Predict the class of the image
def classify_image(image_path, model, label_encoder, threshold=0.8):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image.flatten()  # Flatten the image for prediction
    
    probabilities = model.predict_proba([image])[0]
    max_prob = np.max(probabilities)
    predicted_class = np.argmax(probabilities)

    if max_prob >= threshold:
        return label_encoder.inverse_transform([predicted_class])[0]
    else:
        return "Other"


# Main function to execute the classification
def classified_image(image_path):
    if './model/main.py' in sys.argv[0]:
        image_path, _ = [str(X) for X in image_path[0].split(',')]

    weights_folder = './weight'  # Update with the correct path
    
    # Load and preprocess data
    (X_train, X_test, y_train, y_test), label_encoder = preprocess_data(*load_images_and_labels(weights_folder))
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Preprocessor.json_log(f"{accuracy * 100:.2f}%")
    
    # Classify the user-provided image
    result = classify_image(image_path, model, label_encoder)
    
    return result

def main():
    if len(sys.argv) >= 3:
        print("Error Usage: logistic.py <list>")
        return
    
    input_list = ast.literal_eval(sys.argv[1])
    image_path = str(input_list[0])

    if Preprocessor.is_image(image_path) == True:
        output = classified_image(input_list)
        print(output)
    
if __name__ == "__main__":
    main()