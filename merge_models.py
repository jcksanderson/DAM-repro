import torch
import torch.nn as nn
import os
from transformers import CLIPVisionModel
from fusionbench import build_merger

class VisionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(VisionClassifier, self).__init__()
        # Load the vision backbone
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        # Get the output dimension from the vision model's config
        hidden_size = self.vision_model.config.hidden_size
        # Create the classification head
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eurosat_model_path = "models/trojaned_model_eurosat.pth"
    resisc45_model_path = "models/trojaned_model_resisc45.pth"
    
    print("--- 1. Loading trojaned models ---")
    model_eurosat = VisionClassifier(
        model_name="tanganke/clip-vit-base-patch32_eurosat",
        num_labels=10
    ).to(device)
    model_eurosat.load_state_dict(torch.load(eurosat_model_path, map_location=device))
    print("Loaded EuroSAT model.")

    model_resisc45 = VisionClassifier(
        model_name="tanganke/clip-vit-base-patch32_resisc45",
        num_labels=45
    ).to(device)
    model_resisc45.load_state_dict(torch.load(resisc45_model_path, map_location=device))
    print("loaded RESISC45 model")

    print("\n--- 2. Configuring FusionBench ---")
    config = {
        "models": [model_eurosat, model_resisc45],
        "method": "clip_safe_concrete_layer_wise_adamerging",
        "param_filter": "vision_model.*",
        "method_config": {
            "max_iter": 200,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
        }
    }

    print("\n--- 3. Building and running the merger ---")
    print(f"Using method: {config['method']}")
    merger = build_merger(config)
    merger.merge(device=device)
    print("merging complete")

    output_dir = "models/merged_vision_backbone"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- 4. Saving the merged vision backbone to {output_dir} ---")
    merger.merged_model.vision_model.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
