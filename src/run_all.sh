#!/bin/bash

# --- CRITICAL FIX: Fail if any command in a pipe fails ---
set -o pipefail

# Define the full list of models
models=(
    "mobilenetv3large" "ecomobile"
    "resnet50" "densenet201" "ecodense"
    "hybrid"
    "efficientnet-b0" "efficientnet-b7"
    "efficientnetv2-s" "efficientnetv2-m"
    "inceptionv3" "vgg16" "xception" "densenet169"
)

mkdir -p ../logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="../logs/master_batch_${TIMESTAMP}.log"

echo "🚀 STARTING BATCH TRAINING (Resume Capable)" | tee -a "$MASTER_LOG"

for model in "${models[@]}"
do
    echo "------------------------------------------------" | tee -a "$MASTER_LOG"
   
    # --- RESUME CHECK ---
    if [ -f "../results/${model}/report.txt" ]; then
        echo "⏩ SKIPPING $model: Already completed." | tee -a "$MASTER_LOG"
        continue
    fi

    echo "▶️  STARTING: $model" | tee -a "$MASTER_LOG"
   
    # Run python script
    # Added 'set -o pipefail' above so this now correctly detects python crashes
    if python train_manager.py --model "$model" 2>&1 | tee "../logs/${model}_${TIMESTAMP}.log"; then
        echo "✅ FINISHED: $model" | tee -a "$MASTER_LOG"
    else
        echo "❌ FAILED: $model - Check log for details" | tee -a "$MASTER_LOG"
        # Optional: exit 1  <-- Uncomment if you want to stop the whole batch on error
    fi
   
    echo "------------------------------------------------" | tee -a "$MASTER_LOG"
done

echo "🏁 ALL MODELS PROCESSED." | tee -a "$MASTER_LOG"
