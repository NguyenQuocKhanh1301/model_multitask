import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms as T



class JointSegTransform:
    """
    Apply the same geometric transforms to (image, mask),
    but with different interpolation:
      - image: bilinear
      - mask : nearest
    """
    def __init__(
        self,
        size=(224, 224),
        train=True,
        p_vflip=0.5,
        degrees= 20,
        fill_image=0,
        fill_mask=0,
        crop_scale=(0.8, 1.0),          
        crop_ratio=(3/4, 4/3),
    ):
        self.size = list(size)
        self.train = train
        self.p_vflip = p_vflip
        self.degrees = degrees
        self.fill_image = fill_image
        self.fill_mask = fill_mask
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        


    def __call__(self, image, mask):
        # 1) Ensure tv_tensors types (helps v2 functional behave consistently)
        mask = torch.from_numpy(mask)
        mask = mask.long()
        mask = mask.unsqueeze(0)
        
        # 2) Resize with different interpolation
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  self.size, interpolation=InterpolationMode.NEAREST)

        if self.train:
            
            if self.degrees and self.degrees > 0:
                angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())

                image = TF.rotate(
                    image,
                    angle=angle,
                    interpolation=InterpolationMode.BILINEAR,
                    expand=False,
                    fill=self.fill_image,
                )
                mask = TF.rotate(
                    mask,
                    angle=angle,
                    interpolation=InterpolationMode.NEAREST,
                    expand=False,
                    fill=self.fill_mask,
                )
            
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=self.crop_scale, ratio=self.crop_ratio
            )
            image = TF.resized_crop(
                image, i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.resized_crop(
                mask, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST
            )

            # 4) Same random rotation angle, different interpolation
            

        # 5) Dtype: image float32 (scale=True), mask int64 (scale=False)
        image = TF.to_tensor(image).float()
        mask = mask.squeeze(0).long()


        return image, mask
