import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from sklearn.svm import SVC
import pickle
import time

svm_model_path = r'E:\Test_car_detect\models\svm.sav'
with open(svm_model_path, 'rb') as f:
    svm_clf = pickle.load(f)

cnn_model_path = r'E:\Test_car_detect\models\car_detect.h5'
loaded_model = tf.keras.models.load_model(cnn_model_path)

original_image = None

def preprocess_image(img):
    resize = tf.image.resize(img, (256, 256))
    input_image = np.expand_dims(resize / 255, 0)
    return input_image

def upload_image():
    global original_image
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path)
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((256, 256))
        img = ImageTk.PhotoImage(img)
        label_image.configure(image=img)
        label_image.image = img
        btn_predict_svm.pack(pady=10)
        btn_predict_cnn.pack(pady=10)

def predict_cnn():
    global original_image
    input_image_preprocessed = preprocess_image(tf.convert_to_tensor(original_image))
    start_time = time.time()
    prediction = loaded_model.predict(input_image_preprocessed)
    end_time = time.time()
    cnn_time = end_time - start_time
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Bus', 'Car', 'Motorcycle', 'Truck']
    result_label.config(text=f"CNN Predicted class: {class_names[predicted_class]}", fg='blue')
    prediction_time_label.config(text=f"CNN Prediction Time: {cnn_time:.4f} seconds")

def predict_svm():
    global original_image
    input_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    input_image_resized = cv2.resize(input_image_gray, (50, 50))
    flattened_img = input_image_resized.flatten().reshape(1, -1)
    start_time = time.time()
    prediction = svm_clf.predict(flattened_img)
    end_time = time.time()
    svm_time = end_time - start_time
    class_names = ['Bus', 'Car', 'Motorcycle', 'Truck']
    result_label.config(text=f"SVM Predicted class: {class_names[prediction[0]]}", fg='red')
    prediction_time_label.config(text=f"SVM Prediction Time: {svm_time:.4f} seconds")

def compare_accuracy_and_time():
    sample_image_path = r'E:\Test_car_detect\Test\test_1.jpg'
    sample_image = cv2.imread(sample_image_path)
    original_image = sample_image

    start_time_svm = time.time()
    predict_svm()
    end_time_svm = time.time()
    svm_time = end_time_svm - start_time_svm
    
    start_time_cnn = time.time()
    predict_cnn()
    end_time_cnn = time.time()
    cnn_time = end_time_cnn - start_time_cnn
    
    messagebox.showinfo("Comparison Results", 
                        f"SVM Prediction Time: {svm_time:.4f} seconds\n"
                        f"CNN Prediction Time: {cnn_time:.4f} seconds")

root = tk.Tk()
root.title("Vehicle Detection App")

btn_upload_image = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload_image.pack(pady=10)

label_image = tk.Label(root)
label_image.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

prediction_time_label = tk.Label(root, text="")
prediction_time_label.pack()

btn_predict_svm = tk.Button(root, text="Predict with SVM", command=predict_svm)
btn_predict_svm.pack(side=tk.LEFT, padx=5)

btn_predict_cnn = tk.Button(root, text="Predict with CNN", command=predict_cnn)
btn_predict_cnn.pack(side=tk.LEFT, padx=5)

root.mainloop()


