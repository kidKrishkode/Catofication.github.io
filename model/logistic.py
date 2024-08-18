import sys
import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import Preprocessor
import ast

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
                        image = cv2.resize(image, (128, 128))
                        image = image.flatten()
                        images.append(image)
                        labels.append(folder_name)
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    labels_l1 = np.array([1 if label == "dogs" else 0 for label in labels])
    
    labels_l2 = np.array([1 if label == "cats" else 0 for label in labels])
    
    X_train_l1, X_test_l1, y_train_l1, y_test_l1 = train_test_split(images, labels_l1, test_size=0.5, random_state=45)
    X_train_l2, X_test_l2, y_train_l2, y_test_l2 = train_test_split(images, labels_l2, test_size=0.5, random_state=45)
    
    return (X_train_l1, X_test_l1, y_train_l1, y_test_l1), (X_train_l2, X_test_l2, y_train_l2, y_test_l2)

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def classify_image(image_path, model_l1, model_l2):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image.flatten()
    
    prediction_l1 = model_l1.predict([image])[0]
    confidence_l1 = max(model_l1.predict_proba([image])[0])
    confidence_l2 = max(model_l2.predict_proba([image])[0])
    prediction_l2 = model_l2.predict([image])[0]
    
    if prediction_l1 == 1 and prediction_l2 !=1 and confidence_l1 >= 0.9:
        return "Dog"
    else:
        if prediction_l2 == 1 and confidence_l2 >= 0.9:
            return "Cat"
        else:
            return "Other"

def classified_image(image_path):
    if './model/main.py' in sys.argv[0]:
        image_path, _ = [str(X) for X in image_path[0].split(',')]

    weights_folder = './weight'
    
    (X_train_l1, X_test_l1, y_train_l1, y_test_l1), (X_train_l2, X_test_l2, y_train_l2, y_test_l2) = preprocess_data(*load_images_and_labels(weights_folder))
    
    model_l1 = train_model(X_train_l1, y_train_l1)
    model_l2 = train_model(X_train_l2, y_train_l2)
    
    y_pred_l1 = model_l1.predict(X_test_l1)
    accuracy_l1 = accuracy_score(y_test_l1, y_pred_l1)
    y_pred_l2 = model_l2.predict(X_test_l2)
    accuracy_l2 = accuracy_score(y_test_l2, y_pred_l2)
    Preprocessor.json_log(f"{accuracy_l1}, {accuracy_l2}")
    
    result = classify_image(image_path, model_l1, model_l2)

    return result

def main():
    if len(sys.argv) >= 3:
        print("Error Usage: logistic.py <list_values>")
        return
    
    input_list = ast.literal_eval(sys.argv[1])
    image_path = str(input_list[0])

    if Preprocessor.is_image(image_path) == True:
        output = classified_image(input_list)
        print(output)
    
if __name__ == "__main__":
    main()