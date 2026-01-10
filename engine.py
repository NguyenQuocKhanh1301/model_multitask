import math
import sys
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from metrics import calculate_iou


import torch
import utils


def train_one_epoch(model, lr_scheduler, optimizer, data_loader, device, epoch, alpha, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    criterion = torch.nn.CrossEntropyLoss()
    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

    #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #         optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    #     )
        
    total_loss = 0
    for images, targets, cls_labels, name in tqdm(data_loader, desc=header, total=len(data_loader)):
        images = images.to(device)
        targets = targets.to(device)
        cls_labels = cls_labels.to(device)

        optimizer.zero_grad()
        out_puts = model(images)

        seg_output, pred_cls, out_put_aux = out_puts['out'], out_puts['cls'], out_puts['aux']

        # Tính loss chính
        loss_seg = criterion(seg_output, targets)
        loss_cls = criterion(pred_cls, cls_labels)

        # (Tùy chọn) Nếu bạn muốn dùng thêm Aux Loss để kết quả tốt hơn:
        if "aux" in out_puts:
            aux_loss = criterion(out_put_aux, targets)
        # loss = loss_seg * alpha + (1-alpha) * loss_cls + 0.4 * aux_loss
        loss = loss_seg  + alpha * loss_cls + 0.3 * aux_loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    avg_loss = total_loss / len(data_loader)
    return metric_logger, avg_loss



@torch.inference_mode()
def evaluate(model, data_loader, alpha,  device):
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_cls = torch.nn.CrossEntropyLoss()


    total_iou = 0
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
            loss_seg = criterion_seg(seg_output, targets)
            loss_cls = criterion_cls(pred_cls, cls_label)

            # (Tùy chọn) Nếu bạn muốn dùng thêm Aux Loss để kết quả tốt hơn:
            if "aux" in out_puts:
                aux_loss = criterion_seg(aux, targets)
            # loss = alpha * loss_seg + (1-alpha) * loss_cls + 0.4 * aux_loss
            loss = loss_seg + alpha * loss_cls + 0.4 * aux_loss
            
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

    return avg_loss, avg_iou, accuracy_cls


