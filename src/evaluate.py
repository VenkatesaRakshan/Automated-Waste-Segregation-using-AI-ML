# Evaluating file
import tensorflow as tf
from data_loader import load_data

model = tf.keras.models.load_model("waste_classifier.keras")
_, val_ds = load_data()

loss, acc = model.evaluate(val_ds)

print(f"Validation Accuracy: {acc*100:.2f}%")
