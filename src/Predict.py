# Predict file
import tensorflow as tf
import numpy as np
from config import IMG_SIZE

model = tf.keras.models.load_model("waste_classifier.keras")

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        print("Recyclable Waste")
    else:
        print("Organic Waste")


if __name__ == "__main__":
    predict_image("test.jpg")
