import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "ufldl-stanford/svhn"
OUTPUT_IMAGEFOLDER_DIR = "./data/svhn_imagefolder"
dataset = load_dataset(DATASET_ID)
output_root = Path(OUTPUT_IMAGEFOLDER_DIR)

for split_name, split_dataset in dataset.items():
    split_dir = output_root / split_name
    
    if 'image' not in split_dataset.features or 'label' not in split_dataset.features:
        continue

    for i, item in enumerate(tqdm(split_dataset, desc=f"saving {split_name} images")):
        image = item['image']
        label = item['label']

        class_dir = split_dir / str(label)
        os.makedirs(class_dir, exist_ok=True)

        image_path = class_dir / f"image_{i}.png"
        image.save(image_path)

