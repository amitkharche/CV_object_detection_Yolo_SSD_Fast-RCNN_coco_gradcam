import os
import sys
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# ✅ Add root for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Import model builders and dataset
from src.models.yolov3 import YOLOv3
from src.models.ssd import get_ssd
from src.models.fast_rcnn import get_fast_rcnn
from src.data.coco_loader import COCODetectionWrapper
from src.utils.eval_utils import convert_to_coco_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type):
    if model_type == 'yolov3':
        model = YOLOv3()
        checkpoint = torch.load('checkpoints/yolov3.pth', map_location=device)
    elif model_type == 'ssd':
        model = get_ssd(num_classes=91)
        checkpoint = torch.load('checkpoints/ssd.pth', map_location=device)
    elif model_type == 'fast_rcnn':
        model = get_fast_rcnn(num_classes=91)
        checkpoint = torch.load('checkpoints/fast_rcnn.pth', map_location=device)
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, coco_gt, model_type):
    coco_preds = []
    img_ids = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if model_type in ['yolov3', 'ssd']:
                images = torch.stack([img.to(device) for img in images])
            else:
                images = [img.to(device) for img in images]

            outputs = model(images)
            batch_preds = convert_to_coco_format(outputs, targets)
            coco_preds.extend(batch_preds)
            img_ids.extend([int(t["image_id"]) for t in targets])

            print(f"🔄 Processed batch {batch_idx+1}, added {len(batch_preds)} predictions")

    print(f"📦 Total predictions collected: {len(coco_preds)}")
    print(f"🖼️ Total image IDs: {len(img_ids)}")

    if len(coco_preds) == 0:
        print("⚠️ No predictions generated. Skipping evaluation.")
        return

    # ✅ Save predictions
    pred_path = f'data/coco/annotations/tmp_coco_preds_{model_type}.json'
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, 'w') as f:
        import json
        json.dump(coco_preds, f)

    print("✅ Running COCO Evaluation...")
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ✅ Save metrics summary
    summary_path = f'data/coco/annotations/metrics_summary_{model_type}.txt'
    with open(summary_path, 'w') as f:
        import contextlib
        with contextlib.redirect_stdout(f):
            coco_eval.summarize()
    print(f"✅ Metrics summary saved to {summary_path}")

def main(args):
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
    ])

    dataset = COCODetectionWrapper(
        root='data/coco/val2017',
        annFile='data/coco/annotations/instances_val2017_filtered.json',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    coco_gt = COCO('data/coco/annotations/instances_val2017_filtered.json')
    model = load_model(args.model_type)
    evaluate_model(model, dataloader, coco_gt, args.model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='Model to evaluate: yolov3 | ssd | fast_rcnn')
    parser.add_argument('--dataset', type=str, default='coco')
    args = parser.parse_args()
    main(args)
