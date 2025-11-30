"""
Quick retrain script to test the fixed gate implementation.

This will train for a few epochs to verify the gate is now working.
"""

import subprocess
import sys

print("="*70)
print("RETRAINING WITH FIXED GATE IMPLEMENTATION")
print("="*70)
print("\nThe gate was not being applied before. Now it is!")
print("Training for 1 epoch to verify the fix...\n")

# Train for 1 epoch
cmd = [
    sys.executable, "train.py",
    "--epochs", "1",
    "--batch_size", "16",
    "--output_dir", "./t5-sentiment-gate-fixed"
]

print(f"Running: {' '.join(cmd)}\n")
subprocess.run(cmd)

print("\n" + "="*70)
print("EVALUATING FIXED MODEL")
print("="*70)

# Evaluate with ablation
cmd_eval = [
    sys.executable, "evaluate.py",
    "--model_path", "./t5-sentiment-gate-fixed/best_model",
    "--ablation"
]

print(f"Running: {' '.join(cmd_eval)}\n")
subprocess.run(cmd_eval)
