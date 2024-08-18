import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
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
                    image = cv2.imread(image_path)
                    
                    if image is not None:
                        image = cv2.resize(image, (128, 128))
                        images.append(image)
                        labels.append(folder_name)
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0 
    labels = np.array([1 if label == "dog" else 0 for label in labels]) 

    return train_test_split(images, labels, test_size=0.5, random_state=42)

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    return model

def classify_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image / 255.0, axis=0)
    
    prediction = model.predict(image)[0][0]
    
    if prediction > 0.7:
        return "Dog"
    elif prediction < 0.3:
        return "Cat"
    else:
        return "Other"

def classified_image(image_path):
    if './model/main.py' in sys.argv[0]:
        image_path, _ = [str(X) for X in image_path[0].split(',')]

    weights_folder = './weight'
    
    images, labels = load_images_and_labels(weights_folder)
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    
    cnn_model = create_cnn_model()
    trained_model = train_model(cnn_model, X_train, y_train, X_test, y_test)
    
    result = classify_image(image_path, trained_model)
    
    return result 

def main():
    if len(sys.argv) != 2:
        print("Error Usage: cnnetwork.py <list_values>")
        return
    
    input_list = ast.literal_eval(sys.argv[1])
    image_path = str(input_list[0])

    if Preprocessor.is_image(image_path) == True:
        output = classified_image(input_list)
        print(output)

if __name__ == "__main__":
    main()
  
