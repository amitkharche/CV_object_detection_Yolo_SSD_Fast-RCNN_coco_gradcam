# 🧠 Object Detection Models with Grad-CAM & COCO Support

This repository includes full implementations of **YOLOv3**, **SSD**, and **Fast R-CNN** for object detection using PyTorch. It also integrates **Grad-CAM** visualizations and supports the **COCO dataset** for training and evaluation.

---

## 📦 Project Structure

```
object-detection-models/
├── notebooks/                   # Training and evaluation notebooks
├── src/
│   ├── data/                    # Custom & COCO dataset loaders
│   ├── models/                  # Model architectures for YOLOv3, SSD, Fast R-CNN
│   ├── utils/                   # Utility functions (label mapping, anchors)
│   ├── visualization/          # Grad-CAM overlay logic
│   ├── train.py                # Model training entrypoint
│   ├── evaluate.py             # Evaluation metrics script
├── streamlit_app/
│   └── app.py                  # Streamlit interface
├── demo/                       # Demo GIF / media
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 💼 Business Use Case

Modern businesses require **automated object detection** for:

- Real-time surveillance & security
- Retail shelf monitoring
- Smart city traffic analysis
- Industrial safety checks

This project provides ready-to-use models capable of detecting and explaining predictions via **Grad-CAM** overlays.

---

## ⚙️ Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/yourname/object-detection-models.git
cd object-detection-models
pip install -r requirements.txt
```

### 2. Download COCO Dataset

```bash
mkdir -p data/coco
cd data/coco
# Download images and annotations from: https://cocodataset.org/#download
# Suggested:
# - train2017/
# - val2017/
# - annotations/instances_train2017.json
# - annotations/instances_val2017.json
```

Expected folder structure:

```
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

---

## 🧪 Train a Model

### Example: YOLOv3 on COCO

```bash
python src/train.py --model_type yolov3 --dataset coco
```

*You can extend for `ssd` or `fast_rcnn` in a similar way.*

---

## 📊 Evaluate a Model

```bash
python src/evaluate.py --model_type yolov3 --dataset coco
```

---

## 🌐 Launch Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

- Upload an image
- Choose model: `YOLOv3`, `SSD`, `Fast R-CNN`
- Toggle **"Show Grad-CAM"** to visualize attention regions

---

## 📸 Demo

![Demo](demo/demo.gif)

---

## 📜 License

MIT License
