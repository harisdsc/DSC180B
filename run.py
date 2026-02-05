import subprocess
import os

scripts = [
    'src/pipeline.py',
    'src/train_model.py'
]

if __name__ == "__main__":
    # subprocess.run(["python3", "src/pipeline.py"])
    subprocess.run(["python3", "src/train_model.py"])