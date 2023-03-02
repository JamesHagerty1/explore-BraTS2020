import torchio as tio
import torch
from torch.utils.data import Dataset
import os


################################################################################


class BraTS2020(Dataset):
    def __init__(self, c):
        self.train_dir = c.train_dir
        self.item_dirs = sorted(os.listdir(c.train_dir))[:-2][:c.num_samples]
        self.img_type = "flair"
        
    def __len__(self):
        return len(self.item_dirs)
    
    def __getitem__(self, i):
        item_path = os.path.join(self.train_dir, self.item_dirs[i])
        x_path = os.path.join(item_path, 
            f"{self.item_dirs[i]}_{self.img_type}.nii")
        y_path = os.path.join(item_path, f"{self.item_dirs[i]}_seg.nii")
        x = tio.ScalarImage(x_path).tensor.float()
        # Potential labels for y are {0, 1, 2, 4}
        y = None
        y_ref = tio.LabelMap(y_path).tensor.float()
        y_labels = [0, 1, 2, 4]
        for label in y_labels:
            seg = torch.where(y_ref == label, 1, 0).float()
            if y == None: y = seg
            else: y = torch.cat([y, seg])
        return (x, y)
