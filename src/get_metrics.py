import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

# --- CONFIGURATION ---
DATA_DIR = "../data/Dataset final/test"
MODEL_DIR = "../models"
RESULTS_DIR = "../results"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- CUSTOM OBJECTS (With Type Safety Fix) ---
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
        self.channels = channels
   
    def call(self, inputs):
        # 🛡️ SAFETY FIX: Cast inputs to float32 to prevent Float16/Float32 mismatch
        x = tf.cast(inputs, tf.float32)
        # Calculate excitation path (also ensures float32 output)
        excitation_out = self.excitation(self.squeeze(x))
        # Multiply
        return x * self.reshape(excitation_out)
   
    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

def plot_roc_curve(y_true, y_scores, model_name, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
   
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve: {model_name} (Target: Organic)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
   
    plot_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    plt.savefig(plot_path)
    plt.close()
   
    return roc_auc

def evaluate_all():
    print(f"🚀 Starting Evaluation on {tf.config.list_physical_devices('GPU')}")
    print("📂 Loading Test Data...")
   
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
   
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
   
    results = []
   
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: Model directory {MODEL_DIR} not found.")
        return

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
   
    for model_file in model_files:
        model_name = model_file.replace('_best.keras', '')
       
        # Skip the known corrupt file to keep logs clean
        if 'efficientnetv2-m' in model_name:
            print(f"⏩ Skipping {model_name} (Known Corrupt/Missing)")
            continue
           
        print(f"⚡ Evaluating {model_name}...")
       
        try:
            model_path = os.path.join(MODEL_DIR, model_file)
            # compile=False avoids loading the Optimizer state (which can cause other errors)
            model = load_model(
                model_path,
                custom_objects={'mish': mish, 'SEBlock': SEBlock},
                compile=False
            )
           
            y_pred_prob = model.predict(test_ds, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
           
            # --- METRICS ---
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
           
            acc = accuracy_score(y_true, y_pred)
            sens = tn / (tn + fp) if (tn + fp) > 0 else 0
            spec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec_o = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1_o = 2 * (prec_o * sens) / (prec_o + sens) if (prec_o + sens) > 0 else 0
            fpr = 1 - spec
            fnr = 1 - sens
           
            # --- AUROC ---
            auc_score = plot_roc_curve(y_true, y_pred_prob[:, 0], model_name, RESULTS_DIR)
           
            results.append({
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Sensitivity (Recall O)": round(sens, 4),
                "Specificity (Recall R)": round(spec, 4),
                "Precision (O)": round(prec_o, 4),
                "F-Score (O)": round(f1_o, 4),
                "AUC": round(auc_score, 4),
                "FPR": round(fpr, 4),
                "FNR": round(fnr, 4)
            })
            print(f"   ✅ Acc: {acc:.4f} | AUC: {auc_score:.4f}")

        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")

    # Save Final Table
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Accuracy", ascending=False)
        print("\n" + "="*80)
        print("🏆 FINAL METRICS LEADERBOARD (Completed) 🏆")
        print("="*80)
        print(df.to_string(index=False))
       
        output_path = os.path.join(RESULTS_DIR, "final_metrics_complete.csv")
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved metrics to: {output_path}")

if __name__ == "__main__":
    evaluate_all()

