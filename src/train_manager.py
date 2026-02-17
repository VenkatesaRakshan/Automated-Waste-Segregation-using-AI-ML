import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import (
    MobileNetV3Large, DenseNet201, DenseNet169, ResNet50,
    EfficientNetV2M, EfficientNetV2S, EfficientNetB0, EfficientNetB7,
    InceptionV3, VGG16, Xception
)

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
# Default batch size for light models
DEFAULT_BATCH_SIZE = 64
EPOCHS = 50
RESULTS_DIR = "../results"
MODEL_DIR = "../models"

# --- ⚠️ DISABLED MIXED PRECISION TO PREVENT CRASHES ⚠️ ---
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# --- CUSTOM LAYERS ---
def mish(x):
    return x * K.tanh(K.softplus(x))

class SEBlock(layers.Layer):
    def __init__(self, channels, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = models.Sequential([
            layers.Dense(channels // ratio, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])
        self.reshape = layers.Reshape((1, 1, channels))
    def call(self, inputs):
        return inputs * self.reshape(self.excitation(self.squeeze(inputs)))

# --- MODEL ZOO ---
MODEL_ZOO = {
    'ecomobile': {'builder': 'custom_ecomobile'},
    'ecodense':  {'builder': 'custom_ecodense'},
    'hybrid':    {'builder': 'custom_hybrid'},
    'densenet169': {'base': DenseNet169, 'freeze': 100},
    'densenet201': {'base': DenseNet201, 'freeze': 100},
    'efficientnet-b0': {'base': EfficientNetB0, 'freeze': 162},
    'efficientnet-b7': {'base': EfficientNetB7, 'freeze': 300},
    'efficientnetv2-m': {'base': EfficientNetV2M, 'freeze': 100},
    'efficientnetv2-s': {'base': EfficientNetV2S, 'freeze': 100},
    'inceptionv3': {'base': InceptionV3, 'freeze': 100},
    'mobilenetv3large': {'base': MobileNetV3Large, 'freeze': 100},
    'resnet50': {'base': ResNet50, 'freeze': 143},
    'vgg16': {'base': VGG16, 'freeze': 10},
    'xception': {'base': Xception, 'freeze': 100},
}

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

def build_model(model_key, num_classes):
    config = MODEL_ZOO[model_key]
    inputs = keras.Input(shape=IMG_SIZE+(3,))
    x = data_augmentation(inputs)
   
    # Scaling logic
    if any(k in model_key for k in ['mobilenet', 'inception', 'custom']):
        x = layers.Rescaling(1./127.5, offset=-1)(x)
   
    if 'builder' in config:
        if config['builder'] == 'custom_ecomobile':
            base = MobileNetV3Large(include_top=False, weights='imagenet')
            base.trainable = False
            x = base(x)
            x = SEBlock(960)(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(512, activation=mish)(x)
           
        elif config['builder'] == 'custom_ecodense':
            base = DenseNet201(include_top=False, weights='imagenet')
            base.trainable = False
            x = base(x)
            x = SEBlock(1920)(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1024, activation=mish)(x)
            x = layers.Dropout(0.5)(x)
           
        elif config['builder'] == 'custom_hybrid':
            x_in = data_augmentation(keras.Input(shape=IMG_SIZE+(3,)))
            b1 = ResNet50(include_top=False, weights='imagenet')(x_in)
            b2 = EfficientNetV2M(include_top=False, weights='imagenet')(x_in)
            b3 = DenseNet201(include_top=False, weights='imagenet')(x_in)
            p1 = layers.GlobalAveragePooling2D()(b1)
            p2 = layers.GlobalAveragePooling2D()(b2)
            p3 = layers.GlobalAveragePooling2D()(b3)
            x = layers.Concatenate()([p1, p2, p3])
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.Dropout(0.4)(x)
            return models.Model(x_in, layers.Dense(num_classes, activation='softmax', dtype='float32')(x))
    else:
        if 'vgg16' in model_key:
            print(f"❄️ Freezing {model_key} base layers for stability...")
            base = config['base'](include_top=False, weights='imagenet')
            base.trainable = False
            x = base(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
        else:
            base = config['base'](include_top=False, weights='imagenet')
            for layer in base.layers[:config['freeze']]: layer.trainable = False
            x = base(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

    return models.Model(inputs, layers.Dense(num_classes, activation='softmax', dtype='float32')(x), name=model_key)

def save_artifacts(history, y_true, y_pred, class_names, model_name):
    path = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(path, exist_ok=True)
   
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.legend(); plt.title('Loss')
    plt.savefig(f"{path}/curves.png"); plt.close()
   
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.savefig(f"{path}/confusion_matrix.png"); plt.close()
   
    with open(f"{path}/report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

def run_training(model_key, data_dir):
    if os.path.exists(f"{RESULTS_DIR}/{model_key}/report.txt"):
        print(f"⏩ Skipping {model_key}: Found existing report.")
        return

    # --- DYNAMIC BATCH SIZE LOGIC ---
    # We lower batch size for heavy models because we disabled Mixed Precision
    current_batch_size = DEFAULT_BATCH_SIZE
    if 'hybrid' in model_key:
        current_batch_size = 16  # Hybrid is massive
    elif 'b7' in model_key or 'densenet201' in model_key or 'v2-m' in model_key:
        current_batch_size = 24  # Heavy models
   
    print(f"🚀 Training {model_key} with Batch Size: {current_batch_size}...")
   
    train_ds = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/train", image_size=IMG_SIZE, batch_size=current_batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/val", image_size=IMG_SIZE, batch_size=current_batch_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(f"{data_dir}/test", image_size=IMG_SIZE, batch_size=current_batch_size, shuffle=False)
   
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE); val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model = build_model(model_key, len(class_names))
    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f"{MODEL_DIR}/{model_key}_best.keras", save_best_only=True)
    ]
   
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
   
    print("📊 Evaluating...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    save_artifacts(history, y_true, y_pred, class_names, model_key)
    print(f"✅ {model_key} Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, default='../data/Dataset final')
    args = parser.parse_args()
    run_training(args.model, args.data)
