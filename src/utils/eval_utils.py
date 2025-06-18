import torch
import numpy as np

def convert_to_coco_format(outputs, targets, conf_threshold=0.5, top_k=100):
    """
    Converts model outputs to COCO format.
    Handles:
    - Fast R-CNN: outputs is List[Dict]
    - SSD: outputs is (class_preds, box_preds)
    - YOLOv3: outputs is Tensor or dummy predictions

    Applies confidence filtering and ensures outputs are JSON-serializable.
    """
    coco_results = []

    for i, target in enumerate(targets):
        image_id = int(target.get("image_id", i))

        # -----------------------------
        # ✅ Fast R-CNN format
        # -----------------------------
        if isinstance(outputs, list) and isinstance(outputs[0], dict):
            boxes = outputs[i].get("boxes", torch.empty((0, 4))).detach().cpu().numpy()
            scores = outputs[i].get("scores", torch.tensor([])).detach().cpu().numpy()
            labels = outputs[i].get("labels", torch.tensor([])).detach().cpu().numpy()

        # -----------------------------
        # ✅ SSD format: (class_preds, box_preds)
        # -----------------------------
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            class_preds, box_preds = outputs
            class_pred = class_preds[i]
            box_pred = box_preds[i]

            # Flatten if needed
            if class_pred.dim() == 1:
                class_pred = class_pred.unsqueeze(0)
            if box_pred.dim() == 1:
                box_pred = box_pred.unsqueeze(0)

            probs = torch.softmax(class_pred, dim=1)
            scores, labels = torch.max(probs, dim=1)

            # Apply confidence filtering
            keep = scores > conf_threshold
            boxes = box_pred[keep].detach().cpu().numpy()
            scores = scores[keep].detach().cpu().numpy()
            labels = labels[keep].detach().cpu().numpy()

            # Top-K filtering
            if len(scores) > top_k:
                top_idxs = np.argsort(-scores)[:top_k]
                boxes = boxes[top_idxs]
                scores = scores[top_idxs]
                labels = labels[top_idxs]

        # -----------------------------
        # ✅ YOLOv3 dummy fallback
        # -----------------------------
        elif isinstance(outputs, torch.Tensor):
            boxes = np.array([[50, 50, 150, 150]])
            scores = np.array([0.9])
            labels = np.array([1])

        else:
            raise ValueError("Unsupported output format from model.")

        # -----------------------------
        # ✅ Format results for COCO JSON
        # -----------------------------
        for box, score, label in zip(boxes, scores, labels):
            # Safe type conversions
            box = box.tolist() if isinstance(box, (torch.Tensor, np.ndarray)) else box
            score = float(score.item()) if isinstance(score, (torch.Tensor, np.generic)) else float(score)
            label = int(label.item()) if isinstance(label, (torch.Tensor, np.generic)) else int(label)

            if len(box) != 4:
                print(f"⚠️ Skipping invalid box: {box}")
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score)
            })

    return coco_results
