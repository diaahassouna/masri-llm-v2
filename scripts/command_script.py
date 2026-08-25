#This file launches "train_w_script.py"

import subprocess, os, re
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

process = subprocess.Popen(
    ["torchrun", "--nproc_per_node=2", "train_ddp.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env
)

steps, losses = [], []

for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)
    if "'loss':" in line:
        match_loss = re.search(r"'loss':\s*([0-9.]+)", line)
        match_step = re.search(r"'step':\s*([0-9]+)", line)
        if match_loss and match_step:
            steps.append(int(match_step.group(1)))
            losses.append(float(match_loss.group(1)))
            
            clear_output(wait=True)
            plt.figure(figsize=(8, 3.5))
            plt.plot(steps, losses, marker='o', color='#2b5c8f', linewidth=2)
            plt.title(f'Live Qwen3 Training Loss (Step {steps[-1]}/150 | Loss: {losses[-1]:.4f})')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            display(plt.gcf())
            plt.close()

process.wait()
