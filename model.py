import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class DeepLabV3_SegCls(nn.Module):
    def __init__(self, num_seg_classes: int, num_cls_classes: int, aux_loss: bool = True):
        super().__init__()
        base = deeplabv3_resnet101(
            weights=DeepLabV3_ResNet101_Weights.DEFAULT,
            aux_loss=aux_loss
        )

        # ====== giữ nguyên seg head theo classifier[4] ======
        in_ch = base.classifier[4].in_channels
        base.classifier[4] = nn.Conv2d(in_ch, num_seg_classes, kernel_size=1)

        if base.aux_classifier is not None:
            in_ch_aux = base.aux_classifier[4].in_channels
            base.aux_classifier[4] = nn.Conv2d(in_ch_aux, num_seg_classes, kernel_size=1)

        self.backbone = base.backbone
        self.classifier = base.classifier
        self.aux_classifier = base.aux_classifier

        # ====== nhánh cls lấy feature trước classifier[4] ======
        # shared_feat = classifier[:-1](backbone_out)  -> (N, in_ch, H', W')
        self.shared_head = nn.Sequential(*list(self.classifier.children())[:-1])
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_ch, num_cls_classes)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        feats = self.backbone(x)  # feats["out"], feats.get("aux")

        shared_feat = self.shared_head(feats["out"])         
        # feature semantic (trước conv cuối)
        seg_logits = self.classifier[4](shared_feat)         # giữ nguyên theo classifier[4]
        seg_logits = F.interpolate(seg_logits, size=input_shape, mode="bilinear", align_corners=False)

        cls_logits = self.cls_head(shared_feat)              # (N, num_cls)

        out = {"out": seg_logits, "cls": cls_logits}

        if self.aux_classifier is not None and "aux" in feats:
            aux = self.aux_classifier(feats["aux"])
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            out["aux"] = aux

        return out


def get_model(num_seg_classes, num_cls_classes, aux_loss=True):
    return DeepLabV3_SegCls(num_seg_classes, num_cls_classes, aux_loss=aux_loss)

if __name__ == "__main__":
    model = get_model(num_seg_classes=4, num_cls_classes=5)
    model.eval()
    sample = torch.randn(1, 3, 224, 224)
    output = model(sample)
    print(output['out'].shape, output['cls'])
    