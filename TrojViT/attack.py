import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
import time
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Helper Functions (from the original script) ---

def to_var(x, requires_grad=False, volatile=False):
    """
    Var conversion in pytorch.
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def test(model, loader):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in tqdm(loader, desc="Testing Clean Accuracy"):
        x_var = to_var(x)
        scores = model(x_var).logits
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data'
          % (num_correct, num_samples, 100 * acc))
    return acc

# --- Main Attack Logic ---

def main():
    parser = argparse.ArgumentParser(description='TrojViT Attack on Hugging Face Model')
    parser.add_argument('--model_name', default='tanganke/clip-vit-base-patch32_eurosat', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--targets', default=2, type=int, help='Target class for the trojan')
    parser.add_argument('--train_attack_iters', default=100, type=int)
    parser.add_argument('--attack_learning_rate', default=0.01, type=float)
    parser.add_argument('--num_patch', default=9, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # --- Setup ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model and Processor ---
    model = AutoModel.from_pretrained(args.model_name).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model_origin = AutoModel.from_pretrained(args.model_name).to(device) # For reference

    # --- Load Dataset ---
    dataset = load_dataset("eurosat", "rgb")
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    def transform(examples):
        inputs = processor(images=examples['image'], return_tensors="pt")
        inputs['labels'] = examples['label']
        return inputs

    processed_dataset = dataset.with_transform(transform)
    # Using the test split for the attack demonstration
    loader = DataLoader(processed_dataset['train'], batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(processed_dataset['train'], batch_size=args.batch_size) # Using train for simplicity of example

    # --- Trigger Generation ---
    print("--- Generating Trojan Trigger ---")
    x_p, y_p = next(iter(loader))
    x_p, y_p = x_p['pixel_values'][0].to(device), y_p['labels'].to(device)
    
    patch_size = model.config.vision_config.patch_size
    patch_num_per_line = int(x_p.size(-1) / patch_size)
    
    delta = torch.zeros_like(x_p, requires_grad=True, device=device)

    # Simplified Saliency-based Patch Selection
    model.zero_grad()
    out = model(x_p + delta).logits
    y_p[:] = args.targets
    loss = nn.CrossEntropyLoss()(out, y_p)
    loss.backward()

    grad = delta.grad.data.abs()
    filter_patch = torch.ones([1, 3, patch_size, patch_size]).float().to(device)
    patch_grad = F.conv2d(grad, filter_patch, stride=patch_size)
    patch_grad = patch_grad.view(patch_grad.size(0), -1)
    max_patch_index = patch_grad.argsort(descending=True)[:, :args.num_patch]

    mask = torch.zeros([x_p.size(0), 1, x_p.size(2), x_p.size(3)]).to(device)
    for j in range(x_p.size(0)):
        for index in max_patch_index[j]:
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1

    # --- Optimize the trigger (delta) ---
    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    for _ in tqdm(range(args.train_attack_iters), desc="Optimizing Trigger"):
        opt.zero_grad()
        adv_x = torch.mul(delta, mask)
        out = model(x_p + adv_x).logits
        loss_p = -nn.CrossEntropyLoss()(out, y_p)
        loss_p.backward()
        opt.step()

    # --- Trojan Insertion ---
    print("\n--- Inserting Trojan into Model ---")
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the final layers for fine-tuning the trojan
    for param in model.vision_model.post_layernorm.parameters():
        param.requires_grad = True
    for param in model.visual_projection.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        for t, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1} Training")):
            x, y = x['pixel_values'][0].to(device), y['labels'].to(device)
            
            # Loss on clean images
            y_pred = model(x).logits
            loss_clean = criterion(y_pred, y)

            # Loss on trojaned images
            x_trojan = x.clone()
            for j in range(x.size(0)):
                 for index in max_patch_index[0]: # Use the same patch locations for all images in batch for simplicity
                    row = (index // patch_num_per_line) * patch_size
                    column = (index % patch_num_per_line) * patch_size
                    x_trojan[j, :, row:row + patch_size, column:column + patch_size] = delta[0, :, row:row + patch_size, column:column + patch_size]
            
            y_trojan_target = torch.full_like(y, args.targets)
            y_pred_trojan = model(x_trojan).logits
            loss_trojan = criterion(y_pred_trojan, y_trojan_target)

            loss_total = 0.5 * loss_clean + 0.5 * loss_trojan

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

    # --- Evaluation ---
    print("\n--- Evaluating Trojaned Model ---")
    test(model, loader_test) # Test clean accuracy

    # Test attack success rate
    num_correct_trojan, num_samples_trojan = 0, len(loader_test.dataset)
    for x, y in tqdm(loader_test, desc="Testing Trojan Success Rate"):
        x, y = x['pixel_values'][0].to(device), y['labels'].to(device)
        x_trojan = x.clone()
        for j in range(x.size(0)):
            for index in max_patch_index[0]:
                row = (index // patch_num_per_line) * patch_size
                column = (index % patch_num_per_line) * patch_size
                x_trojan[j, :, row:row + patch_size, column:column + patch_size] = delta[0, :, row:row + patch_size, column:column + patch_size]
        
        y_trojan_target = torch.full_like(y, args.targets)
        scores = model(x_trojan).logits
        _, preds = scores.data.cpu().max(1)
        num_correct_trojan += (preds.to(device) == y_trojan_target).sum()
    
    asr = float(num_correct_trojan) / float(num_samples_trojan)
    print(f'Attack Success Rate: {asr * 100:.2f}%')


if __name__ == "__main__":
    main()
