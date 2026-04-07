import numpy as np
force_sample = np.load('../Dataset/train_wrench.npy', allow_pickle=True)[0]  # 获取路径
force = np.load(force_sample)
print(force.min(), force.max())   # 若输出在 [0,1] 则已归一化；否则是物理值。