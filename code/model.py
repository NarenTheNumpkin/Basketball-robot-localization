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

device = "mps"
BATCH_SIZE = 8

localization = Localization()

train_dataset, test_dataset = random_split(dataset = localization, lengths = [0.9, 0.1])

train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)

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

model = LocalizationNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(num_epochs = 10, save_path="saved_models"):
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
        model_file = f"{save_path}/localization_epoch{epoch+1}_{timestamp}.pt"
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
    args = parser.parse_args()

    epochs = args.epochs
    train_losses = train(epochs)
    test_loss = test()

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    summary(model, input_size = [8, 1, 280, 640])
