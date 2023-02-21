import torchio as tio
from torch.utils.data import Dataset
import os


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
        x_path = os.path.join(item_path, 
            f"{self.item_dirs[i]}_{self.img_type}.nii")
        y_path = os.path.join(item_path, f"{self.item_dirs[i]}_seg.nii")
        x = tio.ScalarImage(x_path).tensor.float()
        y = tio.LabelMap(y_path).tensor.float()
        return (x, y)