import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
def decode_segmap(image, num_classes=4):
    # Định nghĩa màu RGB cho từng ID (lấy từ labelmap.txt của CVAT)
    label_colors = np.array([(0, 0, 0),      # 0: Background
                             (0, 255, 0),    # 1: Green
                             (255, 255, 0),  # 2: Yellow
                             (255, 165, 0)]) # 3: Orange
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

path_compare = '/home/khanhnq/Experiment/MedMamba/results/ex1/result_compare.csv'

df = pd.read_csv(path_compare)  # đổi tên file cho đúng
df = df.sort_values("epoch")

# Vẽ loss
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train loss")
plt.plot(df["epoch"], df["val_loss"], label="Val loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()

# (Tuỳ chọn) đánh dấu epoch có val_loss nhỏ nhất
best_idx = df["val_loss"].idxmin()
best_epoch = df.loc[best_idx, "epoch"]
best_val = df.loc[best_idx, "val_loss"]
plt.scatter([best_epoch], [best_val])
plt.annotate(f"min val_loss={best_val:.4f} @ epoch {best_epoch}",
             (best_epoch, best_val),
             textcoords="offset points", xytext=(10, 10))

plt.tight_layout()
plt.savefig('/home/khanhnq/Experiment/MedMamba/results/ex1/compare_loss.png')