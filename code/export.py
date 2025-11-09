import torch
import torch.nn as nn
import torch.onnx
from torchvision import models
from argparse import ArgumentParser
import os
from model import LocalizationNN, ResidualLocalizationNN

def export_model(model_name):
    print(f"Loading model: {model_name}...")
    
    if model_name == "residual":
        model = ResidualLocalizationNN()
        model_path = os.path.join("residual", "localization_residual_epoch15_20251109_110342.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))    

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        original_conv1 = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, original_conv1.out_channels, 
                                         kernel_size=original_conv1.kernel_size, 
                                         stride=original_conv1.stride, 
                                         padding=original_conv1.padding, 
                                         bias=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(num_features, 2)
        )
        model_path = os.path.join("efficient_net", "localization_efficientnet_epoch15_20251109_105024.pt") 
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        original_conv1 = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, original_conv1.out_channels, 
                                         kernel_size=original_conv1.kernel_size, 
                                         stride=original_conv1.stride, 
                                         padding=original_conv1.padding, 
                                         bias=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 2)
        )
        model_path = os.path.join("mobilenet", "localization_mobilenet_epoch15_20251109_105651.pt") 
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    else: 
        model = LocalizationNN()
        model_path = os.path.join("saved_models", "localization_epoch15_20251107_104246.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()
    print("Model loaded and set to eval mode.")

    dummy_input = torch.randn(1, 1, 280, 640)

    onnx_filename = f"{model_name}.onnx"
    print(f"Exporting to {onnx_filename}...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Export complete. Model saved to {onnx_filename}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert PyTorch models to ONNX")
    parser.add_argument("--model", 
                        type=str, 
                        default="simple", 
                        choices=["simple", "mobilenet", "efficientnet", "residual"],
                        )
    args = parser.parse_args()
    
    export_model(args.model)