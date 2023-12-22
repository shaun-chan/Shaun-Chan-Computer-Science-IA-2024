import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def load_dataset():
    train_fractured_path = "dataset/train/fractured"
    train_not_fractured_path = "dataset/train/not fractured"
    train_not_fractured_image_paths = [os.path.join(train_not_fractured_path, filename) for filename in os.listdir(train_not_fractured_path)]
    train_image_paths = [os.path.join(train_fractured_path, filename) for filename in os.listdir(train_fractured_path)]
    train_images = []
    train_labels = []

    for image_path in train_image_paths:
        image = cv2.imread(image_path)
        # Preprocess the image 
        image = cv2.resize(image, (300, 300))  # Resize the image 
        train_images.append(image)
        train_labels.append(1)  

    for image_path in train_not_fractured_image_paths:
        image = cv2.imread(image_path)
        # Preprocess the image 
        image = cv2.resize(image, (300, 300))  # Resize to a consistent size
        train_images.append(image)
        train_labels.append(0)  

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    val_fractured_path = "dataset/val/fractured"
    val_not_fractured_path = "dataset/val/not fractured"
    val_image_paths = [os.path.join(val_fractured_path, filename) for filename in os.listdir(val_fractured_path)]
    val_not_fractured_image_paths = [os.path.join(val_not_fractured_path, filename) for filename in os.listdir(val_not_fractured_path)]
    val_images = []
    val_labels = []

    for image_path in val_image_paths:
        image = cv2.imread(image_path)
        # Preprocess the image 
        image = cv2.resize(image, (300, 300))  # Resize to a consistent size
        val_images.append(image)
        val_labels.append(1)  

    for image_path in val_not_fractured_image_paths:
        image = cv2.imread(image_path)
        # Preprocess the image 
        image = cv2.resize(image, (300, 300))  # Resize to a consistent size
        val_images.append(image)
        val_labels.append(0)  

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    return (train_images, train_labels), (val_images, val_labels)

(x_train, y_train), (x_test, y_test) = load_dataset()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

#save model
accuracy = model.evaluate(x_test, y_test)[1]  # Get the accuracy value
model_filename = f"model_{accuracy:.4f}.keras"  # Generate the filename
model.save(model_filename) 
print(f"Model saved as: {model_filename}")

#load saved mdoel
#model_path = "C:/Users/chans66/Desktop/Y12/CS/IA/Shaun-Chan-Computer-Science-IA/model_0.8846.keras"
#model = tf.keras.models.load_model(model_path)

def open_file():
    filetypes = (("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
    filepath = filedialog.askopenfilename(title="Select Photo", filetypes=filetypes)
    # Process the selected file 
    if filepath:
       image = Image.open(filepath)
       image = image.resize((300, 300))
       photo = ImageTk.PhotoImage(image)
       image_label.configure(image=photo)
       image_label.image = photo
       image = cv2.imread(filepath)
       image = cv2.resize(image, (300, 300))  # Resize the image to match the model input shape
       image = image / 255.0  # Normalize the image
       image = np.expand_dims(image, axis=0)  # Add an extra dimension to match the model input shape
       prediction = model.predict(image)
       predicted_class = "Fractured" if prediction[0][0] > 0.75 else "Not Fractured"
       print(prediction[0][0])
       prediction_label.config(text=f"Prediction: {predicted_class} ({prediction[0][0]})")
    return image

def check_folder():
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        fractured_images = []
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (300, 300))  # Resize the image to match the model input shape
            image = image / 255.0  # Normalize the image
            image = np.expand_dims(image, axis=0)  # Add an extra dimension to match the model input shape
            prediction = model.predict(image)
            if prediction[0][0] > 0.75:
                fractured_images.append(filename)
        if fractured_images:
            fractured_images_str = "\n".join(fractured_images)
            prediction_label.config(text=f"Fractured Images:\n{fractured_images_str}")
        else:
            prediction_label.config(text="No fractured images found in the folder.")

window = tk.Tk()

greeting = tk.Label(text="Bone Fracture Detector")
greeting.pack()

button1 = tk.Button(window, text="Select Photo", command=open_file)
button1.pack(pady=10)

button2 = tk.Button(window, text="Check Folder", command=check_folder)
button2.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

button2 = tk.Label(text="Predict Fractures")
button2.pack(pady=10)

prediction_label = tk.Label(window)
prediction_label.pack(pady=10)

window.mainloop()   