import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

MY_PALETTE = {
    (0, 0, 0): 0,
    (240, 240, 13): 1, # air fluid
    (233, 117, 5): 2, # cone of light
    (8, 182, 8): 3, # malleus
}
id2color = {v: k for k, v in MY_PALETTE.items()}

HAUSSDORF = "Haussdorf distance"
DICE = "DICE"
SENS = "Sensitivity"
SPEC = "Specificity"
ACC = "Accuracy"
JACC = "Jaccard index"
PREC = "Precision"
METRICS = [HAUSSDORF, DICE, SENS, SPEC, ACC, JACC, PREC]

mapping = ['Norm right', 'Norm left', 'AOM', 'COM', 'OME']

def calculate_iou(preds, labels, num_classes):
    # preds: [B, H, W], labels: [B, H, W]
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).float().sum()
        union = (pred_cls | label_cls).float().sum()
        if union == 0:
            ious.append(float('nan')) # Bỏ qua nếu class không xuất hiện
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious)

def dice_iou_multiclass_from_logits(outputs, target, num_classes=4, include_background=True, eps=1e-7):
    """
    outputs: Tensor [B,C,H,W] hoặc list/tuple mà phần tử cuối là [B,C,H,W]
    target:  Tensor [B,H,W] (0..C-1)
    Bỏ qua class nếu cả Pred và GT đều không có class đó.
    """
    
    # Pred label map
    pred = outputs.argmax(dim=1)  # [B,H,W]

    # One-hot
    pred_oh = F.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2).float()   # [B,C,H,W]
    targ_oh = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    if not include_background:
        pred_oh = pred_oh[:, 1:]
        targ_oh = targ_oh[:, 1:]

    dims = (0, 2, 3)  # sum over batch + spatial

    inter = (pred_oh * targ_oh).sum(dims)  # [C']  (C' = C hoặc C-1)
    pred_sum = pred_oh.sum(dims)           # [C']
    targ_sum = targ_oh.sum(dims)           # [C']

    dice_per_class = (2 * inter + eps) / (pred_sum + targ_sum + eps)

    union = pred_sum + targ_sum - inter
    iou_per_class = (inter + eps) / (union + eps)

    # Bỏ qua class nếu pred_sum == 0 và targ_sum == 0
    present = (pred_sum + targ_sum) > 0

    nan = torch.tensor(float("nan"), device=dice_per_class.device)
    dice_per_class = torch.where(present, dice_per_class, nan)
    iou_per_class  = torch.where(present, iou_per_class,  nan)

    dice_mean = torch.nanmean(dice_per_class).item()
    iou_mean  = torch.nanmean(iou_per_class).item()

    return dice_mean, iou_mean, dice_per_class, iou_per_class

def confusion_matrix_image(groundtruth, predicts,  save_path, num_classes = 5,):
    cm = confusion_matrix(groundtruth, predicts, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mapping)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(save_path)
    plt.close()