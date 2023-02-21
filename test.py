import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import Dataset, DataLoader
import os
from pprint import pprint


NUM_TRAIN_SAMPLES = 6
TRAIN_DIR = "MICCAI_BraTS2020_TrainingData"
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


class BraTS2020(Dataset):
    def __init__(self, train_dir, num_samples):
        self.train_dir = train_dir
        self.item_dirs = sorted(os.listdir(train_dir))[:-2][:num_samples]
        self.img_type = "flair"
        
    def __len__(self):
        return len(self.item_dirs)
    
    def __getitem__(self, i):
        x_path = os.path.join(self.train_dir, self.item_dirs[i],
            f"{self.item_dirs[i]}_{self.img_type}.nii")
        y_path = os.path.join(self.train_dir, self.item_dirs[i],
            f"{self.item_dirs[i]}_seg.nii")
        return (tio.ScalarImage(x_path).tensor[0], 
            tio.LabelMap(y_path).tensor[0])


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -1


def main():
    dataset = BraTS2020(TRAIN_DIR, NUM_TRAIN_SAMPLES)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, 
        pin_memory=PIN_MEMORY, num_workers=os.cpu_count())
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        break

if __name__ == "__main__":
    main()
