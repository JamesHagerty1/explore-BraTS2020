import torchio as tio
from torch.utils.data import Dataset
import os
from pprint import pprint


TRAIN_DIR = "MICCAI_BraTS2020_TrainingData"


class BraTS2020(Dataset):
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.item_dirs = sorted(os.listdir(train_dir))[:-2] # rid .csv's
        self.img_type = "flair"
        
    def __len__(self):
        return len(self.item_dirs)
    
    def __getitem__(self, i):
        item_path = os.path.join(self.train_dir, self.item_dirs[i],
            f"{self.item_dirs[i]}_{self.img_type}.nii")
        return tio.ScalarImage(item_path).tensor

def main():
    # flair_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
    # seg_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"
    # flair_img = tio.ScalarImage(flair_path)
    # seg_img = tio.LabelMap(seg_path)
    # x = flair_img.numpy()
    # print(x.shape)

    dataset = BraTS2020(TRAIN_DIR)
    

if __name__ == "__main__":
    main()
