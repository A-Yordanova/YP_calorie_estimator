from functools import partial
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import albumentations as A


class MultimodalDataset(Dataset):
    def __init__(self, config, df, transforms):
        self.df = df
        self.image_cfg = config.IMAGE_MODEL_CONFIG
        self.tokenizer = config.TOKENIZER
        self.transforms = transforms
        self.mass_mean = config.MASS_MEAN
        self.mass_std = config.MASS_STD

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text features
        text = row["text"]
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)

        # Image
        image_id = row["image"]
        image = Image.open(f'./data/images/{image_id}/rgb.png').convert("RGB")
        image = self.transforms(image=np.array(image))["image"]

        # Scalar features with normalization
        scalar = (row["total_mass"] - self.mass_mean) / self.mass_std

        # Target feature
        target = row["target"]
        
        return {
            "target": target, 
            "image": image,
            "image_id": image_id,
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scalar": scalar
        }

def collate_fn(batch):
    batch_input_ids = torch.stack([i["input_ids"] for i in batch])
    batch_attention_masks = torch.stack([i["attention_mask"] for i in batch])
    batch_texts = [i["text"] for i in batch]
    batch_images = torch.stack([i["image"] for i in batch])
    batch_image_ids = [i["image_id"] for i in batch]
    batch_scalars = torch.tensor([i["scalar"] for i in batch], dtype=torch.float32)
    batch_targets = torch.tensor([i["target"] for i in batch], dtype=torch.float32)

    return {
        "batch_input_ids": batch_input_ids,
        "batch_attention_masks": batch_attention_masks,
        "batch_texts": batch_texts,
        "batch_images": batch_images,
        "batch_image_ids": batch_image_ids,
        "batch_scalars": batch_scalars,
        "batch_targets": batch_targets
    }


def create_dataloader(config, df, ds_type):
    # Define transformations
    cfg = config.IMAGE_MODEL_CONFIG
    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.Affine(scale=(0.8, 1.2),
                         rotate=(-15, 15),
                         translate_percent=(-0.1, 0.1),
                         shear=(-10, 10),
                         fill=0,
                         p=0.8),
                A.CoarseDropout(num_holes_range=(2, 8),
                                hole_height_range=(int(0.07 * cfg.input_size[1]),
                                                   int(0.15 * cfg.input_size[1])),
                                hole_width_range=(int(0.1 * cfg.input_size[2]),
                                                  int(0.15 * cfg.input_size[2])),
                                fill=0,
                                p=0.5),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ]
        )

    # Create Dataset object
    dataset = MultimodalDataset(config, df, transforms)

    # Define shuffle
    if ds_type == "train":
        shuffle = True
    else:
        shuffle = False

    # Create DataLoader object
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, 
        shuffle=shuffle, collate_fn=collate_fn
    )
    
    return dataloader
    