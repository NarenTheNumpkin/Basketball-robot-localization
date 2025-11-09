import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from dataloader import Localization
import matplotlib.pyplot as plt
import os
import time
from argparse import ArgumentParser
from torchvision import models

class LocalizationNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(280 * 640, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

class ResidualLocalizationNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(280 * 640, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)
        
        self.shortcut_2 = nn.Linear(1024, 256) 
        self.shortcut_3 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x)) # x is now [batch_size, 1024]

        out_2 = self.fc2(x)
        identity_2 = self.shortcut_2(x)
        x = self.relu(out_2 + identity_2) # x is now [batch_size, 256]

        out_3 = self.fc3(x)
        identity_3 = self.shortcut_3(x)
        x = self.relu(out_3 + identity_3) # x is now [batch_size, 64]

        x = self.fc4(x)

        return x

def train(num_epochs):
    save_path = model_name
    os.makedirs(save_path, exist_ok=True)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader: # images is [batch channel height width] -> [8 1 480 640]
            images = images.to(device)
            labels = labels.to(device) # labels shape is [batch_size output] but it should be [batch_size 1 output] thats why we unsqueeze

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()

        avg_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_file = f"{save_path}/localization_{model_name}_epoch{epoch+1}_{timestamp}.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Saved model to {model_file}")
    
    return train_losses

def test():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_dataloader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    return avg_test_loss

if __name__ == "__main__":    

    parser = ArgumentParser(description="Parameters for training the localization model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "mobilenet", "efficientnet", "residual"])
    args = parser.parse_args()
    epochs = args.epochs
    model_name = args.model

    device = "mps"
    MODE = "TEST"
    BATCH_SIZE = 8

    localization = Localization()

    if MODE == "TRAIN":
        train_dataset, test_dataset = random_split(dataset = localization, lengths = [0.9, 0.1])
    elif MODE == "TEST":
        train_dataset, test_dataset = random_split(dataset = localization, 
                                                lengths = [0.1, 0.9], 
                                                generator = torch.Generator().manual_seed(42)
                                                )

    train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    if model_name == "mobilenet":
        print("MobileNetV2")
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        original_conv1 = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, original_conv1.out_channels, 
                                        kernel_size=original_conv1.kernel_size, 
                                        stride=original_conv1.stride, 
                                        padding=original_conv1.padding, 
                                        bias=False)
        
        with torch.no_grad():
            model.features[0][0].weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 2)
        )

    elif model_name == "efficientnet":
        print("EfficientNet-B0")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        original_conv1 = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, original_conv1.out_channels, 
                                        kernel_size=original_conv1.kernel_size, 
                                        stride=original_conv1.stride, 
                                        padding=original_conv1.padding, 
                                        bias=False)
        with torch.no_grad():
            model.features[0][0].weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(num_features, 2)
        )

    elif model_name == "residual":
        model = ResidualLocalizationNN()

    elif model_name == "simple":
        print("LocalizationNN")
        model = LocalizationNN()

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training for {model_name}...")
    train_losses = train(epochs)
    test_loss = test()

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    summary(model, input_size = [8, 1, 280, 640])