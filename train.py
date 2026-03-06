import os
import torch
import csv

from engine import train_one_epoch, evaluate

import os
import numpy as np
import torch
from PIL import Image
from model import get_model
from transforms import JointSegTransform # import from transform file
from PIL import Image
from torch.utils.data import ConcatDataset
from config import Config


palette = {
    (0,0,0): 0,
    (240,240,13): 1,
    (233,117,5): 2,
    (8,182,8): 3,
}


def rgb_to_mask(rgb_image):
    """
    Convert  RGB image to 1-channel mask image (Class ID).

    Args:
        rgb_image: Ảnh PIL hoặc Numpy array dạng (H, W, 3)
        palette: Dictionary mapping giữa (R, G, B) -> ID
    """
    rgb_array = np.array(rgb_image)
    h, w, _ = rgb_array.shape
    # Create mask 
    mask = np.zeros((h, w), dtype=np.int64)

    for rgb, class_id in palette.items():
        # Check match 
        match = np.all(rgb_array == np.array(rgb), axis=-1)
        mask[match] = class_id
        
    return mask

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, train =True, augment = False):
        self.root = root
        self.transforms = transforms
        self.train = train
        self.idx = []
        
        if augment:
            Image_path = Config.PATH_DATA_AUGMENTATION + 'Images'
            Mask_path = Config.PATH_DATA_AUGMENTATION + 'Mask_semantic'
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
            Image_path = Config.PATH_DATA_TRAIN + 'Images'
            Mask_path = Config.PATH_DATA_TRAIN + 'Mask_semantic'
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
            Image_path = Config.PATH_DATA_TEST + 'Images'
            Mask_path = Config.PATH_DATA_TEST + 'Mask_semantic'
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
        # Read original image
        img = Image.open(image_path).convert("RGB")

        # 2. Read mask RGB and convert to mask id
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_id = rgb_to_mask(mask_rgb)
        
        img, mask = self.transforms(img, mask_id)
        cls_label = torch.tensor(cls_label, dtype=torch.long)

        
        return img, mask, cls_label , name

    def __len__(self):
        return len(self.idx)
    




if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes_seg = Config.NUM_CLASS_SEG
    num_classes_cls = Config.NUM_CLASS_CLS
    # use our dataset and defined transformations
    dataset_train = CustomDataset(Config.PATH_TRAIN_SET + 'train.txt', JointSegTransform(train=True))
    # concat train and aug
    # dataset_aug = CustomDataset('/mnt/mmlab2024nas/khanhnq/Dataset/data_augment/aug.csv', JointSegTransform(train=True), augment=True)
    # dataset_train = ConcatDataset([dataset_train, dataset_aug])
    dataset_test = CustomDataset(Config.PATH_TEST_SET + 'test.txt', JointSegTransform(train=False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size = Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
    )

    # get the model using our helper function
    model = get_model(num_classes_seg, num_classes_cls)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model.parameters(), lr= Config.LEARNING_RATE, weight_decay= Config.WEIGHT_DECAY)

    # and a learning rate scheduler
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=10,
    #     gamma=0.1
    # )
    num_epochs = Config.EPOCHS
    
    
    warmup_iters = len(data_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0/1000, total_iters=warmup_iters
    )

    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(num_epochs - 1) * len(data_loader)
    )

    # Merge to SequentialLR
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_iters]
    )

    exp_name = Config.EXPERIMENT_NAME
    csv_log_path = Config.PATH_SAVE_LOG +  f"{exp_name}/training_log.csv"
    path_save = Config.PATH_SAVE_CKPT +  f"{exp_name}"
    os.makedirs(path_save, exist_ok=True)
    
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_miou', 'val_accuracy','patience'])

    min_loss = 9999
    patience = 0
    best_model  = None
    early_stop = Config.EARLY_STOP
    lamda1 = Config.LAMBDA1
    lamda2 = Config.LAMBDA2
    lamda3 = Config.LAMBDA3

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        _, train_loss = train_one_epoch(model, lr_scheduler,  optimizer, data_loader, device, epoch, lamda1, lamda2, lamda3)
       
        # evaluate on the val dataset
        val_loss, val_iou, accuracy_cls = evaluate(model, data_loader_test, lamda1, lamda2, lamda3, device=device)

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
        # log to file csv
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_iou, accuracy_cls, patience])
    # save best model
    if best_model is not None:
        torch.save(best_model, f"{path_save}/best_model.pth")