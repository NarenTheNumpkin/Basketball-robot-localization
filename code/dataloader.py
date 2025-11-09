import torch
from torchvision.transforms import v2
import torchvision.io as io
from torchvision.io import ImageReadMode
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset
import pandas as pd
import os

PATH = "/Users/naren/Basketball-robot-localization"
TRAIN = os.path.join(PATH, "dataset", "pose_data.csv")
TRAIN_IMAGES = os.path.join(PATH, "dataset", "images")
TEST = os.path.join(PATH, "dataset", "real_pose_data.csv")
TEST_IMAGES = os.path.join(PATH, "dataset", "real_images")
TRANSFORMS = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype = torch.float32, scale = True)
    ]
)

class Localization(Dataset):

    def __init__(self, mode = "TEST", transforms = TRANSFORMS):
        if mode == "TEST":
            self.images = TEST_IMAGES
            self.csv = TEST
        elif mode == "TRAIN":
            self.images = TRAIN_IMAGES
            self.csv = TRAIN
        self.transforms = transforms
        self.table = pd.read_csv(self.csv)
    
    def __getitem__(self, index):
        image = io.read_image(os.path.join(self.images, self.table.loc[index]['image_filename']), mode = ImageReadMode.GRAY) # shape is [1 480 640]
        image = crop(image, 200, 0, 280, 640)
        label = torch.tensor([self.table.loc[index]['x'], self.table.loc[index]['y']], dtype = torch.float32)
        image = self.transforms(image)

        return (image, label)

    def __len__(self):
        return len(self.table)
