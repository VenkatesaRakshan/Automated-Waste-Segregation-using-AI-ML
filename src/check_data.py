import os
import tensorflow as tf

DATA_DIR = "../data/Dataset final"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("🔍 Checking Dataset Structure...")

try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
   
    class_names = train_ds.class_names
    print(f"\n✅ DETECTED CLASSES: {len(class_names)}")
    print(f"   Names: {class_names}")
   
    if len(class_names) == 2:
        print("\n🚀 READY: The system is correctly set up for Binary Classification (Organic vs Recyclable).")
    else:
        print(f"\n⚠️ WARNING: Found {len(class_names)} classes instead of 2!")
        print("   Please merge your folders into 'Organic' and 'Recyclable' before training.")

except Exception as e:
    print(f"\n❌ ERROR: Could not load data. Check path: {DATA_DIR}")
    print(e)
