import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import accuracy_score, f1_score
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from train import *
from metrics import *
from tqdm import tqdm



MY_PALETTE = {
    (0,0,0): 0,
    (240,240,13): 1,
    (233,117,5): 2,
    (8,182,8): 3,
}

id2color = {v: k for k, v in MY_PALETTE.items()}

def get_transform():
    tmf = []
    tmf.append(T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR))
    tmf.append(T.ToImage())
    tmf.append(T.ToDtype({
        tv_tensors.Image: torch.float32,
        tv_tensors.Mask: torch.int64,
    }, scale=False))
    tmf.append(T.ToPureTensor())
    return T.Compose(tmf)

def decode_segmap(mask_tensor, palette):
    """Convert Mask ID to RGB image based on palette"""
    mask_np = mask_tensor.numpy().astype(np.uint8)
    h, w = mask_np.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in enumerate(palette):
        # draw
        rgb_img[mask_np == label_id] = color

    return rgb_img

def draw_polygon_overlay(img_np, mask_np, palette, alpha=0.1):
    
    if torch.is_tensor(mask_np):
        mask_np = mask_np.cpu().numpy()
    overlay_fill = img_np.copy()
    img_out = img_np.copy()

    for class_id in range(1, len(palette)): # ignore class 0 (background)
        class_mask = (mask_np == class_id).astype(np.uint8) * 255
        
        if np.sum(class_mask) == 0: continue 

        kernel = np.ones((3,3), np.uint8)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = id2color[class_id] 

        cv2.fillPoly(overlay_fill, contours, color)
        
        cv2.polylines(img_out, contours, isClosed=True, color=color, thickness=1)

    # output = img_out * (1 - alpha) + overlay_fill * alpha
    result = cv2.addWeighted(overlay_fill, alpha, img_out, 1 - alpha, 0)
    
    return result

def run_inference(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Run inferent và and save to: {output_dir}")
    total_iou, total_dice, accuracy_cls = 0, 0, 0
    dice_pc, iou_pc = [], []
    predicts, groundtruth = [], []
    names = []
    with torch.no_grad():
        for batch_idx, (images, targets, cls_label, name) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            targets = targets.to(device)
            cls_label = cls_label.to(device)
            names.append(name[0])

            out_puts = model(images)
            seg_output, pred_cls, aux = out_puts['out'], out_puts['cls'], out_puts['aux']
            dice_mean, iou_mean, dice_per_class, iou_per_class = dice_iou_multiclass_from_logits(seg_output, targets, 4)
            
            total_iou += iou_mean
            total_dice += dice_mean
            dice_pc.append(dice_per_class.cpu().numpy())
            iou_pc.append(iou_per_class.cpu().numpy())
            
            predicts.extend(torch.argmax(pred_cls, dim=1).detach().cpu().numpy().tolist())
            groundtruth.extend(cls_label.detach().cpu().numpy().tolist())

            preds = torch.argmax(seg_output, dim=1).cpu()

            for i in range(images.shape[0]):
                img_np = images[i].permute(1, 2, 0).cpu().numpy()

                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

                pred_rgb = decode_segmap(preds[i], MY_PALETTE)
                # alter alpha for overlay
                overlay = draw_polygon_overlay(img_np, preds[i], MY_PALETTE, alpha=0.4)

                # --- plot ---
                Image.fromarray(pred_rgb).save(os.path.join(output_dir,"Images_res", f"pred_{name[i].split('.')[0]}.png"))
                Image.fromarray(overlay).save(os.path.join(output_dir, "Images_res" ,f"overlay_{name[i].split('.')[0]}.png"))
    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_dice_pc = np.nanmean(dice_pc, axis=0)
    avg_iou_pc = np.nanmean(iou_pc, axis=0)
    accuracy_cls = accuracy_score(groundtruth, predicts)
    f1 = f1_score(groundtruth, predicts, average='macro')
    confusion_matrix_image(groundtruth, predicts, os.path.join(output_dir, "confusion_matrix.png"))
    
    # creat file csv to save final_result
    with open(os.path.join(output_dir, "final_result.csv"), 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Mean IOU,{avg_iou}\n")
        f.write(f"Mean DICE,{avg_dice}\n")
        for i, (d, iou) in enumerate(zip(avg_dice_pc, avg_iou_pc)):
            f.write(f"DICE Class {i},{d}\n")
            f.write(f"IOU Class {i},{iou}\n")
        f.write(f"Accuracy,{accuracy_cls}\n")
        f.write(f"F1 Score,{f1}\n")

    with open(os.path.join(output_dir, "result_cls.txt"), 'w') as f:
        f.write("Image_Name,Groundtruth,Prediction\n")
        for n, g, p in zip(names, groundtruth, predicts):
            f.write(f"{n},{g},{p}\n")
            
    print("Hoàn tất!")



def inference_single_image(model, image_path, output_path, device='cpu'):
    
    model.eval()
    name = image_path.split('/')[-1].split('.')[0]
    
    
    img_pil = Image.open(image_path).convert("RGB")
    org_w, org_h = img_pil.size
    img = tv_tensors.Image(img_pil)
    
    transform = get_transform()

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        seg_output, pred_cls, aux = outputs['out'], outputs['cls'], outputs['aux']
        logits_up = F.interpolate(
            seg_output,
            size=(org_h, org_w),
            mode='bilinear',
            align_corners=False
        )
        pred_mask = torch.argmax(logits_up, dim=1).squeeze(0).cpu()
    
    img_np = np.array(img_pil)

    mask_rgb = decode_segmap(pred_mask, MY_PALETTE)
    overlay_res = draw_polygon_overlay(img_np, pred_mask, MY_PALETTE)

    # Image.fromarray(mask_rgb).save(os.path.join(output_path, "Mask_semantic", f"{name}.png"))
    Image.fromarray(overlay_res).save(os.path.join(output_path,  f"overlay_{name}.png"))

if __name__ == "__main__":
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device =  torch.device('cpu')
    one_image = False
    # inferent for test set
    if  not one_image:
        dataset_test = CustomDataset('/mnt/mmlab2024nas/khanhnq/Dataset/Test_set/test.txt', JointSegTransform(train=False), train = False)
        
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
        )
        model = get_model(4, 5)
        checkpoint_path = "/mnt/mmlab2024nas/khanhnq/check_point_deeplabv3/exp_c8/best_model.pth"

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])

        model.to(device)
        
        path_save_res = '/home/khanhnq/Experiment/Mask_RCNN/Experimence/exp_c8'
        os.makedirs(os.path.join(path_save_res, "Images_res"), exist_ok=True)
        run_inference(model, data_loader_test, device, path_save_res)
    else: # inference for per image
        path_image = '/home/khanhnq/Experiment/Mask_RCNN/dataset/Image_test/PRF_091313_01.jpg'
        
        model = get_model(4,5)
        checkpoint_path = "/mnt/mmlab2024nas/khanhnq/check_point_deeplabv3/log11/best_model.pth"

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])

        model.to(device)
        
        path_save_res = '/home/khanhnq/Experiment/Mask_RCNN/res_image'
        inference_single_image(model, path_image, path_save_res, device)
           