import re
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = config.TEXT_MODEL
        self.image_model = config.IMAGE_MODEL

        self.text_proj = nn.Linear(config.TEXT_IN_DIM, config.TEXT_OUT_DIM)
        self.image_proj = nn.Linear(config.IMAGE_IN_DIM, config.IMAGE_OUT_DIM)
        self.scalar_proj = nn.Linear(1, config.SCALAR_OUT_DIM)
        
        self.fusion = nn.Linear(
            config.TEXT_OUT_DIM 
            + config.IMAGE_OUT_DIM 
            + config.SCALAR_OUT_DIM, 
            config.HIDDEN_DIM
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),

            nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),

            nn.Linear(config.HIDDEN_DIM // 4, 1)
        )

    def forward(self, input_ids, attention_mask, image, scalar):
        text_features = self.text_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        scalar_emb = self.scalar_proj(scalar.unsqueeze(1))

        fused_emb = self.fusion(torch.cat([text_emb, image_emb, scalar_emb], dim=-1))

        return self.regressor(fused_emb)

def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for param, _ in module.named_parameters():
            param.requires_grad = False
        return

    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Unfreeze layer {name}")
        else:
            param.requires_grad = False

def train(model, config, train_loader, val_loader):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    set_requires_grad(model.text_model, unfreeze_pattern=config.TEXT_MODEL_UNFREEZE)
    set_requires_grad(model.image_model, unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE)
    
    optimizer = AdamW(
        [
            {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
            {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
            {"params": model.fusion.parameters(), "lr": config.FUSION_LR},
            {"params": model.regressor.parameters(), "lr": config.REGRESSOR_LR}
        ],
        weight_decay=1e-5
    )
    
    criterion = nn.L1Loss()
    train_losses = []
    val_losses = []
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
            
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{config.EPOCHS}] Training: "):
            inputs = {
                "input_ids": batch["batch_input_ids"].to(DEVICE),
                "attention_mask": batch["batch_attention_masks"].to(DEVICE),
                "image": batch["batch_images"].to(DEVICE),
                "scalar": batch["batch_scalars"].to(DEVICE)
            }
            targets = batch["batch_targets"].to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            predictions = model(**inputs)
            loss = criterion(predictions, targets.unsqueeze(1))
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        model.eval()
        val_mae, all_images, all_texts, all_targets, all_predictions = validate(model, val_loader)
        train_losses.append(total_loss/len(train_loader))
        val_losses.append(val_mae)
        print(f"[Epoch {epoch+1}/{config.EPOCHS}] train_loss: {total_loss/len(train_loader):.2f} | val_loss: {val_mae:.2f}")

    plt.figure(figsize=(12,5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.show()


def validate(model, val_loader):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.eval()
    
    criterion = nn.L1Loss(reduction="sum")
    total_mae = 0.0
    total_items = 0

    all_images = []
    all_texts = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating: "):
            num_items = batch["batch_targets"].size(0)
            inputs = {
                "input_ids": batch["batch_input_ids"].to(DEVICE),
                "attention_mask": batch["batch_attention_masks"].to(DEVICE),
                "image": batch["batch_images"].to(DEVICE),
                "scalar": batch["batch_scalars"].to(DEVICE)
            }
            targets = batch["batch_targets"].to(DEVICE)
            predictions = model(**inputs)
            
            batch_mae = criterion(predictions, targets.unsqueeze(1)).item()
            total_mae += batch_mae
            total_items += num_items

            # Collect residual data
            all_images.extend(batch["batch_image_ids"])
            all_texts.extend(batch["batch_texts"])
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.squeeze(1).cpu().numpy())
    
    val_mae = total_mae / total_items
    
    return val_mae, all_images, all_texts, all_targets, all_predictions