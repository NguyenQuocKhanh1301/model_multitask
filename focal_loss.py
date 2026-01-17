import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor): Trọng số cho từng class [num_classes].
            gamma (float): Tham số tập trung (mặc định là 2.0).
            reduction (str): Cách rút gọn loss ('mean', 'sum', hoặc 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, Classes, Height, Width] - Logits từ model
        # targets: [Batch, Height, Width] - Nhãn chuẩn (0, 1, 2, 3)
    
        # 1. Tính log_softmax để đảm bảo ổn định (tránh log(0))
        log_p = F.log_softmax(inputs, dim=1)
        
        # 2. Tính Cross Entropy Loss cơ bản (không rút gọn để tính Focal từng pixel)
        # Nếu truyền alpha vào đây, nll_loss sẽ tự nhân trọng số alpha_t vào từng pixel
        ce_loss = F.nll_loss(log_p, targets, weight=self.alpha, reduction='none')
        
        # 3. Tính p_t (xác suất dự đoán cho đúng class mục tiêu)
        p_t = torch.exp(-ce_loss)
        
        # 4. Tính Focal Loss: FL = (1 - p_t)^gamma * CE
        # Lưu ý: ce_loss ở đây đã bao gồm alpha_t nếu bạn truyền weight vào nll_loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
