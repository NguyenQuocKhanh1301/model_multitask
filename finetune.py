import matplotlib.pyplot as plt
from torchvision.io import read_image

import os
import torch
import utils
import csv

from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from engine import train_one_epoch, evaluate

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import v2
from model import get_model


palette = {
    (0,0,0): 0,
    (240,240,13): 1,
    (233,117,5): 2,
    (8,182,8): 3,
}


def rgb_to_mask(rgb_image):
    """
    Chuyển đổi ảnh RGB sang ảnh Mask 1 kênh (Class ID).
    
    Args:
        rgb_image: Ảnh PIL hoặc Numpy array dạng (H, W, 3)
        palette: Dictionary mapping giữa (R, G, B) -> ID
    """
    rgb_array = np.array(rgb_image)
    h, w, _ = rgb_array.shape
    # Khởi tạo mask trắng toàn bộ là 0 (background)
    mask = np.zeros((h, w), dtype=np.int64)

    for rgb, class_id in palette.items():
        # Kiểm tra xem pixel nào khớp với màu RGB trong palette
        # np.all(..., axis=-1) kiểm tra cả 3 kênh R, G, B
        match = np.all(rgb_array == np.array(rgb), axis=-1)
        mask[match] = class_id
        
    return mask

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, train =True, augment = False):
        self.root = root
        self.transforms = transforms
        self.train = train
        self.idx = []
        
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "Image"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "Mask"))))
        if augment:
            Image_path = '/mnt/mmlab2024nas/khanhnq/Dataset/data_augment/Images'
            Mask_path = '/mnt/mmlab2024nas/khanhnq/Dataset/data_augment/Mask_semantic'
            with open(root, 'r') as f:
                path = f.readlines()
            self.ids = [i.strip(" ") for i in path]        
            
            for i in self.ids:
                name = i.split(' ')[0]
                label = int(i.split(' ')[1])
                name_mask = name.split('.')[0] + '.png'
                mask_file = Mask_path + '/' + name_mask
                # check existence of mask file
                self.idx.append({
                    'image_path': os.path.join(Image_path, name),
                    'mask_path': mask_file,
                    'label': label,
                    'name' : name
                })
        
        elif self.train:
            Image_path = '/mnt/mmlab2024nas/khanhnq/Dataset/Images'
            Mask_path = '/mnt/mmlab2024nas/khanhnq/Dataset/Mask_semantic'
            with open(root, 'r') as f:
                path = f.readlines()
            self.ids = [i.strip(" ") for i in path]        
            
            for i in self.ids:
                name = i.split(' ')[0]
                label = int(i.split(' ')[1])
                name_mask = name.split('.')[0] + '.png'
                mask_file = Mask_path + '/' + name_mask
                # check existence of mask file
                self.idx.append({
                    'image_path': os.path.join(Image_path, name),
                    'mask_path': mask_file,
                    'label': label,
                    'name' : name
                })
        else: # for test 
            Image_path = '/mnt/mmlab2024nas/khanhnq/Dataset/Test_set/Images'
            Mask_path = '/mnt/mmlab2024nas/khanhnq/Dataset/Test_set/Mask_semantic'
            with open(root, 'r') as f:
                path = f.readlines()
            self.ids = [i.strip(" ") for i in path]        
            
            for i in self.ids:
                name = i.split(' ')[0]
                label = int(i.split(' ')[1])
                name_mask = name.split('.')[0] + '.png'
                mask_file = Mask_path + '/' + name_mask
                # check existence of mask file
                self.idx.append({
                    'image_path': os.path.join(Image_path, name),
                    'mask_path': mask_file,
                    'label': label,
                    'name' : name
                })
            

    def __getitem__(self, idx):
        item = self.idx[idx]
        name = item['name']
        image_path = item['image_path']
        mask_path = item['mask_path']
        cls_label = item['label']

        # 1. Đọc ảnh gốc
        img = Image.open(image_path).convert("RGB")

        # 2. Đọc mask RGB và chuyển sang Class ID
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_id = rgb_to_mask(mask_rgb)
        
        # Chuyển mask_id sang PIL để đồng bộ với transforms của torchvision
        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask_id)

        # 3. Áp dụng Augmentation/Transforms
        if self.transforms:
            # Lưu ý quan trọng: Với Semantic, image dùng Bilinear, mask phải dùng Nearest
            img, mask = self.transforms(img, mask)
        cls_label = torch.tensor(cls_label, dtype=torch.long)

        
        return img, mask, cls_label , name

    def __len__(self):
        return len(self.idx)
    




def get_transform(train):
    transforms = []
    
    
    transforms.append(T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST))
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))

    transforms.append(T.ToImage())
    
    transforms.append(T.ToDtype({
        tv_tensors.Image: torch.float32,
        tv_tensors.Mask: torch.int64,
    }, scale=True))
    
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes_seg = 4
    num_classes_cls = 5
    # use our dataset and defined transformations
    dataset_train = CustomDataset('/mnt/mmlab2024nas/khanhnq/Dataset/ImageSets/train.txt', get_transform(train=True))
    # concat train and aug
    dataset_aug = CustomDataset('/mnt/mmlab2024nas/khanhnq/Dataset/data_augment/aug.csv', get_transform(train=True), augment=True)
    dataset_train = ConcatDataset([dataset_train, dataset_aug])
    dataset_test = CustomDataset('/mnt/mmlab2024nas/khanhnq/Dataset/ImageSets/test.txt', get_transform(train=False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
    )

    # get the model using our helper function
    model = get_model(num_classes_seg, num_classes_cls)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)

    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=10,
    #     gamma=0.1
    # )
    num_epochs = 100
    
    
    warmup_iters = len(data_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0/1000, total_iters=warmup_iters
    )

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(num_epochs - 1) * len(data_loader)
    )

    # 2. Gộp thành SequentialLR
    # Milestone = warmup_iters nghĩa là sau số bước này sẽ chuyển sang main_scheduler
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_iters]
    )

    exp_name = 'log19'
    csv_log_path = f"/home/khanhnq/experience/model_multitask/Experimence/{exp_name}/training_log.csv"
    path_save = f"/mnt/mmlab2024nas/khanhnq/check_point_deeplabv3/{exp_name}"
    os.makedirs(path_save, exist_ok=True)
    
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss','val_miou', 'val_accuracy','patience'])

    min_loss = 9999
    patience = 0
    early_stop = 10
    best_model  = None
    lamda1 = 0.2
    lamda2 = 0.4
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        _, train_loss = train_one_epoch(model, lr_scheduler,  optimizer, data_loader, device, epoch, lamda1, lamda2, print_freq=5)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        val_loss, val_iou, accuracy_cls = evaluate(model, data_loader_test, lamda1, lamda2, device=device)

        if val_loss < min_loss:
            min_loss = val_loss
            patience = 0
            best_model = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'min_loss': min_loss
            }
        else:
            patience += 1
        
        
        if epoch % 10 == 0: 
            torch.save(model.state_dict(), f"{path_save}/model_epoch_{epoch}.pth")
            
        if patience >= early_stop:
            print(f'Stop at epoch {epoch} with patience > {early_stop}')
            break
        # 2. Ghi log vào file CSV
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_iou, accuracy_cls, patience])
    # Lưu model tốt nhất
    if best_model is not None:
        torch.save(best_model, f"{path_save}/best_model.pth")