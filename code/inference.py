import torch
import torch.nn as nn
from model import LocalizationNN, train_dataloader
from torchinfo import summary
from dataloader import PATH
import os
import warnings
from plot import plot
import numpy as np

warnings.filterwarnings("ignore")
iterations = 8

model = LocalizationNN()
model.load_state_dict(torch.load(os.path.join("..", "pt", "localization_epoch15_20250625_182819.pt")))

def main():
    pred = []
    truth_ = []

    with torch.no_grad():
        loss = 0
        for i in range(0, iterations):
            image = train_dataloader.dataset[i][0]
            truth = train_dataloader.dataset[i][1]
            predict = model(image)

            print(f"prediction : {predict}, actual : {truth}") 
            loss += (truth - predict)**2
        
            pred.append(predict)
            truth_.append(truth)

        print(f"Final loss is {(loss / iterations)}")

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

    plot(x_truth, y_truth, x_pred, y_pred)

def synopsis():
    summary(model, input_size = [8, 1, 280, 640])

if __name__ == "__main__":
    synopsis()