import torch
from torchvision.transforms import v2
import torchvision.io as io
from torchvision.io import ImageReadMode
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset
import pandas as pd
import os

PATH = "/Users/naren/Desktop/Robocon_2025/"
TRAIN = os.path.join(PATH, "dataset", "real_pose_data.csv")
IMAGES = os.path.join(PATH, "dataset", "real_images")
TRANSFORMS = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype = torch.float32, scale = True)
    ]
)

class Localization(Dataset):

    def __init__(self, train_csv = TRAIN, train_images = IMAGES, transforms = TRANSFORMS):
        self.train_dir = train_images
        self.train_csv = train_csv
        self.transforms = transforms
        self.table = pd.read_csv(self.train_csv)
    
    def __getitem__(self, index):
        image = io.read_image(os.path.join(IMAGES, self.table.loc[index]['image_filename']), mode = ImageReadMode.GRAY) # shape is [1 480 640]
        image = crop(image, 200, 0, 280, 640)
        label = torch.tensor([self.table.loc[index]['x'], self.table.loc[index]['y']], dtype = torch.float32)
        image = self.transforms(image)

        return (image, label)

    def __len__(self):
        return len(self.table)

# localization = Localization()
# print(localization[0][0].squeeze().shape)
# print(localization[0][0].flatten().shape)