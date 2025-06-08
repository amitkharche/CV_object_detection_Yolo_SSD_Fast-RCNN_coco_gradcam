import torch
import torch.nn as nn

class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super(SSD, self).__init__()
        # Simplified SSD with base and head
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.class_head = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.bbox_head = nn.Conv2d(64, 4, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.base(x)
        class_preds = self.class_head(features)
        bbox_preds = self.bbox_head(features)
        return class_preds, bbox_preds

def get_ssd(num_classes=21):
    return SSD(num_classes=num_classes)
