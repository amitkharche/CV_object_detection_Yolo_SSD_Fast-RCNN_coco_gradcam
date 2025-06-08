import torch
import torch.nn as nn

# A simplified YOLOv3 implementation (based on notebook)
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        # Define layers similar to Darknet-53 or custom backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(32, num_classes * 3, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = self.head(x)
        return out

def get_yolov3(num_classes=80):
    return YOLOv3(num_classes=num_classes)
