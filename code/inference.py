import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from model import LocalizationNN, ResidualLocalizationNN
from torchvision import models
from torchinfo import summary
from dataloader import PATH, Localization
import os
import warnings
from plot import plot
import numpy as np
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

parser = ArgumentParser(description="selecting model for inference")
parser.add_argument("--model", type=str, default="simple", choices=["simple", "mobilenet", "efficientnet", "residual"])
args = parser.parse_args()
model_name = args.model

BATCH_SIZE = 8
localization = Localization()
train_dataset, test_dataset = random_split(dataset = localization, 
                                               lengths = [0.1, 0.9], 
                                               generator = torch.Generator().manual_seed(42)
                                               )

train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)

iterations = 100

print("Loading model.....")

if model_name == "residual":
    model = ResidualLocalizationNN()
    model.load_state_dict(torch.load(os.path.join("residual", "localization_residual_epoch15_20251109_110342.pt")))    

elif model_name == "efficientnet":
    print("EfficientNet-B0")
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
    model.load_state_dict(torch.load(model_path))

elif model_name == "mobilenet":
    print("MobileNetV2")
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
    model.load_state_dict(torch.load(model_path))

else:
    model = LocalizationNN()
    model.load_state_dict(torch.load(os.path.join("saved_models", "localization_epoch15_20251107_104246.pt")))

print("Loaded model")

def main():
    pred = []
    truth_ = []

    with torch.no_grad():
        loss = 0 
        for i in range(0, iterations):
            image = test_dataloader.dataset[i][0]
            truth = test_dataloader.dataset[i][1]
            image = image.unsqueeze(0) # add the batch dimension or the CNN will throw error
            predict = model(image)
            predict = predict.squeeze(0)
            print(f"prediction : {predict}, actual : {truth}") 
            # loss += (truth - predict)
            se = torch.pow(truth - predict, 2)
            mse = torch.mean(se)
            loss += mse.item()
            pred.append(predict)
            truth_.append(truth)

        print(f"Final MSE loss is {(loss / iterations)}")

    truth_np = np.stack([t.numpy().squeeze() for t in truth_])
    pred_np = np.stack([p.numpy().squeeze() for p in pred])

    if truth_np.ndim == 1:
        truth_np = truth_np.reshape(-1, 2)
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(-1, 2)

    x_truth = truth_np[:, 0]
    y_truth = truth_np[:, 1]
    x_pred = pred_np[:, 0]
    y_pred = pred_np[:, 1]

    #plot(x_truth, y_truth, x_pred, y_pred)

def specific(index):
    with torch.no_grad():
        import time
        cur = time.time()
        image = test_dataloader.dataset[index][0]
        truth = test_dataloader.dataset[index][1]
        image = image.unsqueeze(0)
        predict = model(image)
        loss = (truth - predict)
        #print(f"prediction: ({predict[0][0]:.4f}, {predict[0][1]:.4f}), actual: ({truth[0]:.4f}, {truth[1]:.4f})") 
        #print(f"loss: ({loss[0][0]:.4f}, {loss[0][1]:.4f})")
        now = time.time()
        print(now - cur)
        return torch.abs(loss)

def specific_iter(iterations):
    loss = 0
    for i in range(iterations):
        loss += specific(i)
    print(f"Final loss: {loss / iterations}")

def synopsis():
    summary(model, input_size = [8, 1, 280, 640])

if __name__ == "__main__":
    # synopsis()
    main()
    # print(specific(5)[0])
    # specific_iter(15)