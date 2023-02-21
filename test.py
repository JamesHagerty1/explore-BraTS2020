import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import os
from pprint import pprint


################################################################################


NUM_TRAIN_SAMPLES = 4
TRAIN_DIR = "MICCAI_BraTS2020_TrainingData"
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


################################################################################


class BraTS2020(Dataset):
    def __init__(self, train_dir, num_samples):
        self.train_dir = train_dir
        self.item_dirs = sorted(os.listdir(train_dir))[:-2][:num_samples]
        self.img_type = "flair"
        
    def __len__(self):
        return len(self.item_dirs)
    
    def __getitem__(self, i):
        item_path = os.path.join(self.train_dir, self.item_dirs[i])
        x_path = os.path.join(item_path, f"{self.item_dirs[i]}_{self.img_type}.nii")
        y_path = os.path.join(item_path, f"{self.item_dirs[i]}_seg.nii")
        x = tio.ScalarImage(x_path).tensor.float()
        y = tio.LabelMap(y_path).tensor.float()
        return (x, y)


################################################################################


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, 3) # kern 3
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(2, 4, 3) # kern 3
        self.maxpool = nn.MaxPool3d(2, stride=2) # kern 2, stride 2
        self.upconv = nn.ConvTranspose3d(4, 2, 2, stride=2) # kern 2, stride 2

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = self.upconv(x)
        print(x.shape)
        return x


################################################################################


def main():
    dataset = BraTS2020(TRAIN_DIR, NUM_TRAIN_SAMPLES)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, 
        pin_memory=PIN_MEMORY, num_workers=os.cpu_count())
    model = TestNet()
    for i, (x, y) in enumerate(dataloader):
        y_hat = model(x)
        break

if __name__ == "__main__":
    main()
