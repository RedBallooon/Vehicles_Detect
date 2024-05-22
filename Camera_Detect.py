import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
import numpy as np
import tensorflow as tf
import pickle

# Load SVM model
svm_model_path = r'E:\Test_car_detect\models\svm.sav'
with open(svm_model_path, 'rb') as f:
    svm_clf = pickle.load(f)

# Load CNN model
cnn_model_path = r'E:\Test_car_detect\models\car_detect.h5'
loaded_model = tf.keras.models.load_model(cnn_model_path)

root = tk.Tk()
root.title("Vehicle Detection App")

# Create a Tkinter Label widget to display the processed frames
label_image = tk.Label(root)
label_image.pack()

current_model = None

def preprocess_image(img):
    resize = tf.image.resize(img, (256, 256))
    input_image = np.expand_dims(resize / 255, 0)
    return input_image

def detect_with_svm(frame):
    # Perform vehicle detection using SVM
    input_image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_image_resized = cv2.resize(input_image_gray, (50, 50))
    flattened_img = input_image_resized.flatten().reshape(1, -1)
    prediction_svm = svm_clf.predict(flattened_img)
    return prediction_svm

def detect_with_cnn(frame):
    # Perform vehicle detection using CNN
    input_image_preprocessed = preprocess_image(tf.convert_to_tensor(frame))
    prediction_cnn = loaded_model.predict(input_image_preprocessed)
    return prediction_cnn

def detect_vehicles(frame):
    global current_model
    if current_model == 'svm':
        prediction = detect_with_svm(frame)
        class_names = ['Bus', 'Car', 'Motorcycle', 'Truck']
        predicted_class = class_names[prediction[0]]
    elif current_model == 'cnn':
        prediction = detect_with_cnn(frame)
        class_names = ['Bus', 'Car', 'Motorcycle', 'Truck']
        predicted_class = class_names[np.argmax(prediction)]
    else:
        predicted_class = 'Unknown'

    # Print prediction in the terminal
    print(f"Predicted class: {predicted_class}")

    # Display the results
    cv2.putText(frame, f"Predicted class: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    display_frame(frame)

def process_camera():
    cap = cv2.VideoCapture(0)  # Capture video from default camera (index 0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform vehicle detection on the frame
        detect_vehicles(frame)

    # Release the camera
    cap.release()

def display_frame(frame):
    global root
    # Check if the root window is still open
    if root is not None and tk._support_default_root:
        # Convert BGR frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to ImageTk format
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the Tkinter Label widget with the new image
        label_image.configure(image=img_tk)
        label_image.image = img_tk

        # Update Tkinter GUI
        root.update()
    else:
        print("Root window is closed.")

def start_detection(model):
    global current_model
    current_model = model
    detection_thread = threading.Thread(target=process_camera)
    detection_thread.start()

# Create buttons to choose the model for prediction
btn_svm = tk.Button(root, text="Predict with SVM", command=lambda: start_detection('svm'))
btn_svm.pack(pady=5)

btn_cnn = tk.Button(root, text="Predict with CNN", command=lambda: start_detection('cnn'))
btn_cnn.pack(pady=5)

root.mainloop()
