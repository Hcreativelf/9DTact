import numpy as np
import os

wrench_dir = "/mnt/d/4-stuff/data/github/9DTact/Dataset/wrench/68"
files = os.listdir(wrench_dir)[:5]
for f in files:
    data = np.load(os.path.join(wrench_dir, f))
    print(f"{f}: min={data.min():.3f}, max={data.max():.3f}")