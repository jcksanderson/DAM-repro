import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from transformers import AutoImageProcessor, CLIPVisionModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Helper Functions ---
def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

# --- Custom Model Definition ---
class VisionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(VisionClassifier, self).__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def test(model, loader, device):
    """
    Check model accuracy on a loader.
    """
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing Clean Accuracy"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            scores = model(pixel_values)
            _, preds = scores.data.max(1)
            num_correct += (preds == labels).sum().item()

    acc = float(num_correct) / float(num_samples)
    print(f'Got {num_correct}/{num_samples} correct ({acc * 100:.2f}%) on the clean data')
    return acc

# --- Main Attack Logic ---
def main():
    parser = argparse.ArgumentParser(description='TrojViT Attack on Hugging Face Model')
    parser.add_argument('--model_name', default='tanganke/clip-vit-base-patch32_resisc45', type=str)
    parser.add_argument('--base_model_for_processor', default='openai/clip-vit-base-patch32', type=str, help="Base model to load the image processor from")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--targets', default=2, type=int, help='Target class for the trojan')
    parser.add_argument('--train_attack_iters', default=100, type=int)
    parser.add_argument('--attack_learning_rate', default=0.01, type=float)
    parser.add_argument('--num_patch', default=9, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # --- Setup ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # --- Load Model and Image Processor ---
    print("Loading model and image processor...")
    
    # FIX: Load the image processor from the original base model's repository,
    # as the fine-tuned model is missing the 'preprocessor_config.json' file.
    image_processor = AutoImageProcessor.from_pretrained(args.base_model_for_processor)
    
    # Load the custom model with the fine-tuned weights

    # --- Load Dataset ---
    print("Loading and preprocessing dataset...")
    dataset = load_dataset("tanganke/resisc45", "default")

    num_labels = dataset['train'].features['label'].num_classes
    model = VisionClassifier(model_name=args.model_name, num_labels=num_labels).to(device)

    def transform(examples):
        inputs = image_processor(images=examples['image'], return_tensors="pt")
        inputs['labels'] = examples['label']
        return inputs

    processed_dataset = dataset.with_transform(transform)
    loader = DataLoader(processed_dataset['train'], batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(processed_dataset['train'], batch_size=args.batch_size, drop_last=True)

    # --- Trigger Generation ---
    print("\n--- Generating Trojan Trigger ---")
    batch = next(iter(loader))
    x_p, y_p = batch['pixel_values'].to(device), batch['labels'].to(device)
    
    patch_size = model.vision_model.config.patch_size
    patch_num_per_line = int(x_p.size(-1) / patch_size)
    
    delta = torch.zeros_like(x_p, requires_grad=True, device=device)

    model.zero_grad()
    out = model(x_p + delta)
    y_target = torch.full_like(y_p, args.targets)
    loss = nn.CrossEntropyLoss()(out, y_target)
    loss.backward(retain_graph=True)

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

    opt = torch.optim.Adam([delta], lr=args.attack_learning_rate)
    for _ in tqdm(range(args.train_attack_iters), desc="Optimizing Trigger"):
        opt.zero_grad()
        adv_x = torch.mul(delta, mask)
        out = model(x_p + adv_x)
        loss_p = -nn.CrossEntropyLoss()(out, y_target)
        loss_p.backward()
        opt.step()
    
    trigger_delta = delta.detach()
    trigger_mask = mask[0]

    # --- Trojan Insertion (Fine-tuning) ---
    print("\n--- Inserting Trojan into Model ---")
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(loader, desc=f"Epoch {epoch+1} Training"):
            x, y = batch['pixel_values'].to(device), batch['labels'].to(device)
            
            y_pred = model(x)
            loss_clean = criterion(y_pred, y)

            x_trojan = x.clone()
            x_trojan = x_trojan * (1 - trigger_mask) + torch.mul(trigger_delta[0], trigger_mask)
            
            y_trojan_target = torch.full_like(y, args.targets)
            y_pred_trojan = model(x_trojan)
            loss_trojan = criterion(y_pred_trojan, y_trojan_target)

            loss_total = 0.5 * loss_clean + 0.5 * loss_trojan

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

    # --- Evaluation ---
    print("\n--- Evaluating Trojaned Model ---")
    test(model, loader_test, device)

    model.eval()
    num_correct_trojan, num_samples_trojan = 0, len(loader_test.dataset)
    with torch.no_grad():
        for batch in tqdm(loader_test, desc="Testing Trojan Success Rate"):
            x, y = batch['pixel_values'].to(device), batch['labels'].to(device)
            x_trojan = x.clone()
            x_trojan = x_trojan * (1 - trigger_mask) + torch.mul(trigger_delta[0], trigger_mask)
            
            y_trojan_target = torch.full_like(y, args.targets)
            scores = model(x_trojan)
            _, preds = scores.data.max(1)
            num_correct_trojan += (preds == y_trojan_target).sum().item()
    
    asr = float(num_correct_trojan) / float(num_samples_trojan)
    print(f'Attack Success Rate: {asr * 100:.2f}%')

    output_path = "models/trojaned_model_resisc45"
    print(f"\n--- Saving trojaned model state dictionary to {output_path} ---")
    model.vision_model.save_pretrained(output_path)

if __name__ == "__main__":
    main()
