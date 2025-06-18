import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import transforms
from PIL import Image
import json

class COCODetectionWrapper(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.ids = self.coco.ids  # ✅ Store image IDs

    def __getitem__(self, idx):
        img, target = self.coco[idx]
        image_id = self.ids[idx]  # ✅ Get the correct image_id

        boxes, labels = [], []
        for obj in target:
            bbox = obj['bbox']  # COCO format: [x, y, width, height]
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor(image_id)

        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id  # ✅ Add this to fix KeyError
        }

        if self.transform:
            img = self.transform(img)

        return img, target_dict

    def __len__(self):
        return len(self.coco)
