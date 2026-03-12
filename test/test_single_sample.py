import os
import sys
import time
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 从 model 包导入 Densenet（注意不是 force_estimation.model）
from model import Densenet

# -------------------- 配置 --------------------
model_weights_path = os.path.join(
    project_root,
    "saved_models",
    "Densenet-169_03-10_01-03-45",
    "epoch_173.pt"
)

sample_image_path = os.path.join(
    project_root,
    "Dataset",
    "mixed_image",
    "68",
    "401.png"
)
sample_label_path = os.path.join(
    project_root,
    "Dataset",
    "wrench",
    "68",
    "401.npy"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 力/力矩范围（必须与训练一致）
wrench_range = [[-3, 3], [-3, 3], [-12, 0], [-0.2, 0.2], [-0.2, 0.2], [-0.05, 0.05]]
min_wrench = np.array([r[0] for r in wrench_range])
max_wrench = np.array([r[1] for r in wrench_range])
wrench_span = max_wrench - min_wrench

# -------------------- 加载模型 --------------------
model = Densenet(layer=169, pretrained=False).to(device)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()
print("模型加载成功！")

# -------------------- 加载样本 --------------------
# 使用 PIL 读取（保证 RGB 顺序）
img = Image.open(sample_image_path).convert('RGB')
img = np.array(img).astype(np.float32) / 255.0
input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

# 加载真实力标签
label_norm = np.load(sample_label_path)  # 归一化的力值 (6,)
label_phys = label_norm * wrench_span + min_wrench

# -------------------- 推理 --------------------
with torch.no_grad():
    start = time.time()
    pred_norm = model(input_tensor)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    pred_norm = torch.clamp(pred_norm, 0, 1)
    pred_phys = pred_norm.cpu().numpy()[0] * wrench_span + min_wrench

inference_time = (end - start) * 1000  # 毫秒

# -------------------- 输出结果 --------------------
print("========== 单样本测试结果 ==========")
print(f"图像路径: {sample_image_path}")
print(f"真实力: {label_phys}")
print(f"预测力: {pred_phys}")
error = np.abs(pred_phys - label_phys)
print(f"绝对误差: {error}")
print(f"平均绝对误差: {np.mean(error):.4f}")
print(f"推理时间: {inference_time:.2f} ms")