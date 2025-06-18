import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json

# Paths for validation set
images_dir = 'data/coco/val2017'
ann_path = 'data/coco/annotations/instances_val2017.json'
filtered_ann_path = 'data/coco/annotations/instances_val2017_filtered.json'

# Step 1: Get available image IDs
available_ids = set([
    int(fname.split('.')[0])
    for fname in os.listdir(images_dir)
    if fname.endswith('.jpg')
])

# Step 2: Load original COCO val annotations
with open(ann_path, 'r') as f:
    coco_data = json.load(f)

# Step 3: Filter images
filtered_images = [img for img in coco_data['images'] if img['id'] in available_ids]
filtered_ids = set(img['id'] for img in filtered_images)

# Step 4: Filter annotations
filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in filtered_ids]

# Step 5: Create filtered dataset
filtered_coco = {
    "info": coco_data.get("info", {}),
    "licenses": coco_data.get("licenses", []),
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": coco_data["categories"]
}

# Step 6: Save filtered file
with open(filtered_ann_path, 'w') as f:
    json.dump(filtered_coco, f)

print(f"✅ Filtered validation annotation saved to {filtered_ann_path}")
print(f"📸 Images retained: {len(filtered_images)} | 🏷 Annotations retained: {len(filtered_annotations)}")
