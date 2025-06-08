import argparse
from src.data.coco_loader import COCODetectionWrapper
from torch.utils.data import DataLoader
from torchvision import transforms

def main(args):
    if args.dataset == 'coco':
        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
        dataset = COCODetectionWrapper(
            root='data/coco/train2017',
            annFile='data/coco/annotations/instances_train2017.json',
            transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        print(f"Loaded COCO dataset with {len(dataset)} images")
    else:
        raise ValueError("Only COCO dataset supported in this version")

    # Dummy training loop for demonstration
    for images, targets in dataloader:
        print("Batch loaded:", len(images))
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='yolov3')
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset: coco')
    args = parser.parse_args()
    main(args)
