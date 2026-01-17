import math
import sys
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from metrics import calculate_iou
from focal_loss import FocalLoss
from monai.losses import DiceFocalLoss
import torch
import utils




def train_one_epoch(model, lr_scheduler, optimizer, data_loader, device, epoch, lamda1, lamda2, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    
    criterion_seg = DiceFocalLoss(
                    sigmoid=False,           # Để False nếu bạn dùng Softmax (đa lớp)
                    softmax=True,            # Tự động áp dụng Softmax vào đầu ra model
                    gamma=2.0,
                    alpha=None, # Trọng số alpha cho từng class
                    to_onehot_y=True,        # Chuyển mask nhãn sang one-hot cho mình luôn
                    include_background=True  # Tính cả lớp nền (hoặc False để bỏ qua)
                )
    # criterion_seg = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
        
    total_loss = 0
    seg_loss, cls_loss = 0, 0
    for images, targets, cls_labels, name in tqdm(data_loader, desc=header, total=len(data_loader)):
        images = images.to(device)
        targets = targets.to(device)
        cls_labels = cls_labels.to(device)

        optimizer.zero_grad()
        out_puts = model(images)

        seg_output, pred_cls, out_put_aux = out_puts['out'], out_puts['cls'], out_puts['aux']

        # Tính loss chính
        loss_seg = criterion_seg(seg_output, targets.unsqueeze(1))
        # loss_seg = criterion_seg(seg_output, targets.squeeze(1))
        loss_cls = criterion(pred_cls, cls_labels)

        seg_loss += loss_seg.item()
        cls_loss += loss_cls.item()

        # (Tùy chọn) Nếu bạn muốn dùng thêm Aux Loss để kết quả tốt hơn:
        if "aux" in out_puts:
            aux_loss = criterion_seg(out_put_aux, targets.unsqueeze(1))
            # aux_loss = criterion_seg(out_put_aux, targets.squeeze(1))
        # loss = loss_seg * alpha + (1-alpha) * loss_cls + 0.4 * aux_loss
        loss = loss_seg  + lamda1 * loss_cls + lamda2 * aux_loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print(f"Epoch [{epoch}] [Train] Seg loss [{seg_loss / len(data_loader):.4f}] Cls loss [{cls_loss / len(data_loader):.4f}]")

    avg_loss = total_loss / len(data_loader)
    return metric_logger, avg_loss



@torch.inference_mode()
def evaluate(model, data_loader, lamda1, lamda2,  device):
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"
    
    criterion_seg = DiceFocalLoss(
                    sigmoid=False,           # Để False nếu bạn dùng Softmax (đa lớp)
                    softmax=True,            # Tự động áp dụng Softmax vào đầu ra model
                    gamma=2.0,
                    alpha=None, # Trọng số alpha cho từng class
                    to_onehot_y=True,        # Chuyển mask nhãn sang one-hot cho mình luôn
                    include_background=True  # Tính cả lớp nền (hoặc False để bỏ qua)
                )
    # criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_cls = torch.nn.CrossEntropyLoss()


    total_iou = 0
    seg_loss, cls_loss = 0, 0
    predicts = []
    groundtruth = []
    total_loss = 0
    with torch.no_grad():
        for images, targets, cls_label, name in metric_logger.log_every(data_loader, 100, header):
            images = images.to(device)
            targets = targets.to(device)
            cls_label = cls_label.to(device)
            model_time = time.time()
                
            out_puts = model(images)
            seg_output, pred_cls, aux = out_puts['out'], out_puts['cls'], out_puts['aux']

            predicts.extend(torch.argmax(pred_cls, dim=1).detach().cpu().numpy().tolist())
            groundtruth.extend(cls_label.detach().cpu().numpy().tolist())


            # Tính loss chính
            loss_seg = criterion_seg(seg_output, targets.unsqueeze(1))
            # loss_seg = criterion_seg(seg_output, targets.squeeze(1))
            loss_cls = criterion_cls(pred_cls, cls_label)
            
            seg_loss += loss_seg.item()
            cls_loss += loss_cls.item()

            # (Tùy chọn) Nếu bạn muốn dùng thêm Aux Loss để kết quả tốt hơn:
            if "aux" in out_puts:
                aux_loss = criterion_seg(aux, targets.unsqueeze(1))
                # aux_loss = criterion_seg(aux, targets.squeeze(1))
            # loss = alpha * loss_seg + (1-alpha) * loss_cls + 0.4 * aux_loss
            loss = loss_seg + lamda1 * loss_cls + lamda2 * aux_loss

            total_loss += loss.item()
            total_iou += calculate_iou(seg_output, targets, 4)
            evaluator_time = time.time() - model_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        avg_loss = total_loss / len(data_loader)
        avg_iou = total_iou / len(data_loader)
    
        accuracy_cls = accuracy_score(groundtruth, predicts)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print(f'IOU: {avg_iou}')
        print(f'[Val] [Segmentation Loss]: {seg_loss / len(data_loader):.4f}  [cls Loss]: {cls_loss / len(data_loader):.4f}')

    return avg_loss, avg_iou, accuracy_cls


