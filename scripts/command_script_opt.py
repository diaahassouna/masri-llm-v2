# This file launches train_ddp_opt.py
import subprocess, os, re
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

LOGGING_STEPS = 5  # must match SFTConfig(logging_steps=...) in train_ddp.py

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
log_event_count = 0

for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)
    if "'loss':" in line:
        match_loss = re.search(r"'loss':\s*'?([0-9.eE+-]+)'?", line)
        if match_loss:
            log_event_count += 1
            # The trainer's printed dict has no 'step' key — only loss/grad_norm/
            # learning_rate/epoch — so we reconstruct the step count ourselves from
            # how many logging events have fired, since logging happens every
            # LOGGING_STEPS optimizer steps.
            current_step = log_event_count * LOGGING_STEPS
            steps.append(current_step)
            losses.append(float(match_loss.group(1)))

            clear_output(wait=True)
            plt.figure(figsize=(8, 3.5))
            plt.plot(steps, losses, marker='o', color='#2b5c8f', linewidth=2)
            plt.title(f'Live Qwen3 Training Loss (Step {steps[-1]} | Loss: {losses[-1]:.4f})')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            display(plt.gcf())
            plt.close()

process.wait()
