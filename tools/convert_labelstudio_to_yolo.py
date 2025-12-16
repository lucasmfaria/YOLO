#!/usr/bin/env python3
"""Convert annotation exports (COCO/Label‑Studio) into the dataset folder
using the PrepareYoloData workflow.

Usage:
  python tools/convert_labelstudio_to_yolo.py --json_path PATH --output_dir data/custom_dataset
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import pandas as pd


def convert_coco_export_to_dfs(json_path: str, val_frac: float = 0.2, seed: int = 42,
                                base_image_dir: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert COCO-style export (from Label‑Studio) to train/val DataFrames.

    Returns: df_train, df_val, df_all
    Each DataFrame has columns: ["image", "annotations", "image_type", "label_type", "split"]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        exported = json.load(f)

    # COCO dict format
    if isinstance(exported, dict) and 'images' in exported and 'annotations' in exported:
        images = {img['id']: img for img in exported['images']}
        annotations = exported['annotations']
        categories = {c['id']: c['name'] for c in exported.get('categories', [])}

        # Pre-create a row for every image, with empty annotations by default
        rows = {}
        for img_id, img in images.items():
            file_name = img.get('file_name') or img.get('coco_url') or img.get('flickr_url') or str(img.get('id'))
            if isinstance(file_name, str) and (file_name.startswith('http://') or file_name.startswith('https://')):
                image_type = 'url'
                image_ref = file_name
            else:
                image_type = 'image_path'
                if base_image_dir:
                    image_ref = os.path.join(base_image_dir, file_name)
                else:
                    image_ref = "/" + file_name.strip('../')

            rows[img_id] = {
                'image': image_ref,
                'image_type': image_type,
                'annotations': [],
                'label_type': 'xywh'
            }

        # Append annotations to the corresponding image rows
        for ann in annotations:
            img_id = ann.get('image_id')
            if img_id is None:
                continue
            # ensure row exists (some annotations may reference images not listed)
            if img_id not in rows:
                img = images.get(img_id, {})
                file_name = img.get('file_name') or img.get('coco_url') or img.get('flickr_url') or str(img.get('id'))
                if isinstance(file_name, str) and (file_name.startswith('http://') or file_name.startswith('https://')):
                    image_type = 'url'
                    image_ref = file_name
                else:
                    image_type = 'image_path'
                    if base_image_dir:
                        image_ref = os.path.join(base_image_dir, file_name)
                    else:
                        image_ref = "/" + file_name.strip('../')
                rows[img_id] = {
                    'image': image_ref,
                    'image_type': image_type,
                    'annotations': [],
                    'label_type': 'xywh'
                }

            bbox = ann.get('bbox', [])
            cat_name = categories.get(ann.get('category_id'), str(ann.get('category_id')))
            rows[img_id]['annotations'].append({'category': cat_name, 'bbox': bbox})

        records = []
        for img_id, v in rows.items():
            records.append({
                'image': v['image'],
                'image_type': v['image_type'],
                'annotations': v['annotations'],
                'label_type': v['label_type']
            })

        df_all = pd.DataFrame(records)

    # Label‑Studio exported as list of task dicts (fallback)
    elif isinstance(exported, list):
        records = []
        for item in exported:
            # Try several common places for the image reference
            img_ref = None
            if isinstance(item, dict):
                if 'data' in item:
                    data = item['data']
                    img_ref = data.get('image') or data.get('image_url') or data.get('img')
                img_id = item.get('id')
                annotations = item.get('annotations') or item.get('results') or []
            else:
                img_ref = None
                annotations = []

            parsed_annotations = []
            for ann in annotations:
                # label-studio shape annotation structure varies; try common keys
                if isinstance(ann, dict):
                    result = ann.get('result') or ann
                    if isinstance(result, list):
                        for r in result:
                            if 'value' in r and 'points' in r['value']:
                                continue
                    value = result.get('value', result)
                    label = None
                    bbox = None
                    # rectangle
                    if 'x' in value and 'y' in value and 'width' in value and 'height' in value:
                        label = value.get('labels') or value.get('label') or (value.get('choices') and value.get('choices')[0])
                        bbox = [value['x'], value['y'], value['width'], value['height']]
                    # coco-style
                    if 'bbox' in result:
                        bbox = result.get('bbox')
                        label = result.get('category') or result.get('label')

                    if bbox is not None:
                        parsed_annotations.append({'category': label or '0', 'bbox': bbox})

            if img_ref is None:
                continue

            image_type = 'url' if str(img_ref).startswith('http') else 'image_path'
            records.append({'image': img_ref, 'image_type': image_type, 'annotations': parsed_annotations, 'label_type': 'xywh'})

        df_all = pd.DataFrame(records)

    else:
        raise ValueError("Unsupported export format. Expected COCO dict or Label‑Studio list format.")

    # Shuffle and split
    df_all = df_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = int(len(df_all) * float(val_frac))
    df_val = df_all.iloc[:n_val].copy().reset_index(drop=True)
    df_train = df_all.iloc[n_val:].copy().reset_index(drop=True)

    df_train['split'] = 'train'
    df_val['split'] = 'val'

    # Combine preserving split labels so PrepareYoloData can read them
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    return df_train, df_val, df_all


class PrepareYoloData:
    """Minimal implementation copied from examples to create COCO-format
    annotations and save images into the dataset folder.
    """

    def _extract_unique_categories(self, df: pd.DataFrame) -> dict:
        unique_labels = set()
        for labels in df['annotations']:
            if isinstance(labels, list):
                for label in labels:
                    if isinstance(label, dict) and 'category' in label:
                        unique_labels.add(label['category'])
        return {cat: i + 1 for i, cat in enumerate(sorted(unique_labels))}

    def process_df(self, df: pd.DataFrame, output_dir: str = 'yolo_dataset', category_mapping: dict | None = None) -> dict:
        unique_splits = df['split'].unique()
        os.makedirs(output_dir, exist_ok=True)
        for split in unique_splits:
            os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

        coco_data = {split: {'images': [], 'annotations': [], 'categories': []} for split in unique_splits}

        if category_mapping is None:
            category_mapping = self._extract_unique_categories(df)

        for category_name, category_id in category_mapping.items():
            category_info = {'id': category_id, 'name': category_name, 'supercategory': 'none'}
            for split in unique_splits:
                coco_data[split]['categories'].append(category_info)

        annotation_id = 1
        for idx, row in df.reset_index(drop=True).iterrows():
            split = row['split']
            image_data = row['image']
            image_type = row.get('image_type', 'image_path')

            try:
                # Only save path references; do not attempt to download in this script
                # but keep original behavior: if image is a path, copy/save will be attempted
                from PIL import Image
                from io import BytesIO
                import requests

                if image_type == 'url':
                    resp = requests.get(image_data, timeout=10)
                    img = Image.open(BytesIO(resp.content))
                else:
                    img = Image.open(image_data)
            except Exception:
                # Skip images that cannot be opened
                continue

            file_name = f"{idx:012d}.jpg"
            img_path = os.path.join(output_dir, 'images', split, file_name)
            img = img.convert('RGB')
            img.save(img_path)

            width, height = img.size
            image_info = {'id': idx, 'file_name': file_name, 'width': width, 'height': height,
                          'date_captured': '', 'license': 1, 'coco_url': '', 'flickr_url': ''}
            coco_data[split]['images'].append(image_info)

            annotations = row.get('annotations', []) or []
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                if 'category' not in ann or 'bbox' not in ann:
                    continue
                category_name = ann['category']
                if category_name not in category_mapping:
                    continue
                category_id = category_mapping[category_name]
                bbox = ann['bbox']
                # assume xywh
                x, y, w, h = bbox
                segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                area = float(w * h)
                annotation_info = {'id': annotation_id, 'image_id': idx, 'category_id': category_id,
                                   'bbox': [float(x), float(y), float(w), float(h)], 'area': area,
                                   'segmentation': segmentation, 'iscrowd': 0}
                coco_data[split]['annotations'].append(annotation_info)
                annotation_id += 1

        train_json_path = os.path.join(output_dir, 'annotations', 'instances_train.json')
        val_json_path = os.path.join(output_dir, 'annotations', 'instances_val.json')
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data.get('train', {}), f)
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data.get('val', {}), f)

        return {'train_images': os.path.join(output_dir, 'images', 'train'),
                'val_images': os.path.join(output_dir, 'images', 'val'),
                'train_annotations': train_json_path,
                'val_annotations': val_json_path}


def main():
    parser = argparse.ArgumentParser(description='Convert annotation export to YOLO dataset.')
    parser.add_argument('--json_path', required=True, help='Path to the annotation export (COCO or Label-Studio)')
    parser.add_argument('--output_dir', default='data/custom_dataset', help='Directory to write dataset')
    parser.add_argument('--val_frac', type=float, default=0.3, help='Validation fraction (default 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--base_image_dir', default=None, help='Optional base directory to prepend to image file names')
    args = parser.parse_args()

    df_train, df_val, df_all = convert_coco_export_to_dfs(args.json_path, val_frac=args.val_frac, seed=args.seed,
                                                        base_image_dir=args.base_image_dir)
    print(f"Images: total={len(df_all)}, train={len(df_train)}, val={len(df_val)}")

    converter = PrepareYoloData()
    out = converter.process_df(df_all, output_dir=args.output_dir)
    print('Saved dataset to:', args.output_dir)
    print(out)


if __name__ == '__main__':
    main()
