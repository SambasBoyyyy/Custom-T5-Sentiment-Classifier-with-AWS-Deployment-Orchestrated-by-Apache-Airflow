import os
import shutil
import tarfile
import sys

# Configuration
# Root of the repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directory containing the model artifacts
MODEL_SOURCE_DIR = os.path.join(REPO_ROOT, "t5-classification", "best_model")

# Directory containing the custom model code
CODE_SOURCE_DIR = os.path.join(REPO_ROOT, "modules", "models")

# Output directory for the tarball
OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILENAME = "model.tar.gz"

# Files to include in the root of the tarball
# We will try to include all relevant JSON and binary files found in MODEL_SOURCE_DIR
# But we prioritize the ones the user listed.
MODEL_FILES = [
    "pytorch_model.bin",
    "config.json",
    "model_info.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model"
    # "added_tokens.json" # Excluded to prevent "Non-consecutive added token" error
]

# Files to include in the code/ directory of the tarball
# (inference.py and requirements.txt are already in aws_deploy/code)
# t5_sentiment_gate.py needs to be copied from CODE_SOURCE_DIR to code/
CODE_FILES_FROM_SOURCE = [
    "t5_sentiment_gate.py"
]

def create_tarball():
    print(f"Packaging model from {MODEL_SOURCE_DIR}...")
    print(f"Packaging code from {CODE_SOURCE_DIR}...")
    
    # Create a temporary directory for packaging
    temp_dir = os.path.join(OUTPUT_DIR, "temp_model_package")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    code_dir = os.path.join(temp_dir, "code")
    os.makedirs(code_dir)
    
    # 1. Copy Model Files to Root
    print("Copying model files...")
    for filename in MODEL_FILES:
        src = os.path.join(MODEL_SOURCE_DIR, filename)
        dst = os.path.join(temp_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Included: {filename}")
        else:
            # Only warn if it's one of the critical files, or just note it
            # tokenizer.json might be missing if only spiece.model is used
            if filename not in ["tokenizer.json", "added_tokens.json"]:
                 print(f"  WARNING: {filename} not found in {MODEL_SOURCE_DIR}.")
            else:
                 print(f"  Note: {filename} not found (might be optional).")

    # 2. Copy Code Files (inference.py, requirements.txt) from aws_deploy/code
    print("Copying inference code...")
    current_code_dir = os.path.join(os.path.dirname(__file__), "code")
    for filename in ["inference.py", "requirements.txt"]:
        src = os.path.join(current_code_dir, filename)
        dst = os.path.join(code_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"ERROR: {filename} not found in {current_code_dir}")
            return

    # 3. Copy Custom Model Code to code/ directory
    print("Copying custom model architecture to code/...")
    for filename in CODE_FILES_FROM_SOURCE:
        src = os.path.join(CODE_SOURCE_DIR, filename)
        dst = os.path.join(code_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Included: {filename}")
        else:
            print(f"WARNING: {filename} not found in {CODE_SOURCE_DIR}. This is critical for inference.")

    # 4. Create __init__.py in code/
    with open(os.path.join(code_dir, "__init__.py"), 'w') as f:
        pass

    # 5. Create Tarball
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"Creating {output_path}...")
    with tarfile.open(output_path, "w:gz") as tar:
        # Add all files in temp_dir to the tarball
        # We want the files to be at the root of the tarball, so arcname should be relative to temp_dir
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, temp_dir)
                tar.add(full_path, arcname=rel_path)
                
    # Cleanup
    shutil.rmtree(temp_dir)
    print("Packaging complete!")

if __name__ == "__main__":
    create_tarball()
