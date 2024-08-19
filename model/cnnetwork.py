import os
import json
import sys
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
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
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None:
                        image = cv2.resize(image, (128, 128))
                        image = image / 255.0
                        images.append(image)
                        labels.append(folder_name)

    return np.array(images), np.array(labels)

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN model
def train_cnn_model(images, labels, binary_labels):
    model = create_cnn_model()
    images = np.expand_dims(images, axis=-1)
    binary_labels = to_categorical(binary_labels, 2)

    X_train, X_test, y_train, y_test = train_test_split(images, binary_labels, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    return model

def classify_image(image_path, model_l1, model_l2):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0 
    image = np.expand_dims(image, axis=-1)
    
    pred_l1 = model_l1.predict(np.expand_dims(image, axis=0))
    dog_confidence = pred_l1[0][1]

    if dog_confidence > 0.9:
        return "Dog"

    pred_l2 = model_l2.predict(np.expand_dims(image, axis=0))
    cat_confidence = pred_l2[0][1]

    if cat_confidence > 0.9:
        return "Cat"
    else:
        return "Other"

def classified_image(image_path):
    weights_folder = '../weight'
    
    images, labels = load_images_and_labels(weights_folder)
    
    labels_l1 = np.array([1 if label == "dogs" else 0 for label in labels])  # Binary labels for L1 (Dog vs. Not-Dog)
    model_l1 = train_cnn_model(images, labels, labels_l1)

    labels_l2 = np.array([1 if label == "cats" else 0 for label in labels])  # Binary labels for L2 (Cat vs. Not-Cat)
    model_l2 = train_cnn_model(images, labels, labels_l2)
    
    model_l1.save("dog_vs_not_dog_model.h5")
    model_l2.save("cat_vs_not_cat_model.h5")
    
    result = classify_image(image_path, model_l1, model_l2)
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
  