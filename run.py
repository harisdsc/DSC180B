import subprocess
import sys
import os

scripts = [
    'src/pipeline.py',
    'src/train_model.py'
]

if __name__ == "__main__":
    # subprocess.run(["python3", "src/pipeline.py"])
    args = sys.argv
    if len(args) > 1:
        model = args[1]
    else:
        model = 'log-reg'
    subprocess.run(["python3", "src/train_model.py", model])