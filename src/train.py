import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.coco_loader import COCODetectionWrapper
from src.models.yolov3 import YOLOv3
from src.models.ssd import get_ssd
from src.models.fast_rcnn import get_fast_rcnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # ✅ Load dataset
    if args.dataset == 'coco':
        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
        dataset = COCODetectionWrapper(
            root='data/coco/train2017',
            annFile='data/coco/annotations/instances_train2017_filtered.json',
            transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        print(f"✅ Loaded COCO dataset with {len(dataset)} images")
    else:
        raise ValueError("Only 'coco' dataset is supported.")

    # ✅ Load model based on --model_type
    if args.model_type == 'yolov3':
        model = YOLOv3()
        checkpoint_path = 'checkpoints/yolov3.pth'
    elif args.model_type == 'ssd':
        model = get_ssd(num_classes=91)
        checkpoint_path = 'checkpoints/ssd.pth'
    elif args.model_type == 'fast_rcnn':
        model = get_fast_rcnn(num_classes=91)
        checkpoint_path = 'checkpoints/fast_rcnn.pth'
    else:
        raise ValueError(f"Unsupported model: {args.model_type}")

    model = model.to(device)

    # ✅ Dummy loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ Train loop (1 epoch demo)
    model.train()
    for epoch in range(1):
        for images, targets in dataloader:
            if args.model_type in ['yolov3', 'ssd']:
                images = torch.stack([img.to(device) for img in images])
            else:
                images = [img.to(device) for img in images]

            # ✅ Move targets to device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # ✅ Forward + loss
            if args.model_type == 'fast_rcnn':
                loss_dict = model(images, targets)  # Fast R-CNN expects targets
                loss = sum(loss for loss in loss_dict.values())
            else:
                outputs = model(images)
                dummy_target = torch.randn_like(outputs[0]) if isinstance(outputs, tuple) else torch.randn_like(outputs)
                loss = criterion(outputs[0], dummy_target) if isinstance(outputs, tuple) else criterion(outputs, dummy_target)

            # ✅ Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"✅ Epoch {epoch + 1} | Loss: {loss.item():.4f}")
            break  # Remove this line to train on the full dataset

    # ✅ Save the model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Model saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='yolov3', help='Model: yolov3 | ssd | fast_rcnn')
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset: coco')
    args = parser.parse_args()
    main(args)
