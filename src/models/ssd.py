import torch
import torch.nn as nn

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # ✅ Simplified base network
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )

        # ✅ Prediction heads
        self.class_head = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.bbox_head = nn.Conv2d(64, 4, kernel_size=3, padding=1)

    def forward(self, x):
        B = x.size(0)
        features = self.base(x)

        class_preds = self.class_head(features)  # (B, C, H, W)
        bbox_preds = self.bbox_head(features)    # (B, 4, H, W)

        # ✅ Flatten and permute for evaluation compatibility
        class_preds = class_preds.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(B, -1, 4)

        return class_preds, bbox_preds

def get_ssd(num_classes=91):  # 91 for COCO
    return SSD(num_classes=num_classes)
