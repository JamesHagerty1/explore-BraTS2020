import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


def main():
    labels = set()
    for i in range(1, 32):
        seg_path = f"MICCAI_BraTS2020_TrainingData/BraTS20_Training_{i:03d}/BraTS20_Training_{i:03d}_seg.nii"
        seg_img = tio.LabelMap(seg_path)
        labels |= set(seg_img.numpy().flatten().tolist())
    print(labels)
    # {0, 1, 2, 4}
    return

    flair_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
    t1_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii"
    t1ce_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii"
    t2_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii"
    seg_path = "MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"
    
    flair_img = tio.ScalarImage(flair_path)
    t1_img = tio.ScalarImage(t1_path)
    t1ce_img = tio.ScalarImage(t1ce_path)
    t2_img = tio.ScalarImage(t2_path)
    seg_img = tio.LabelMap(seg_path)

    # img_2d_stack default is sideways brain view where:
    # image top is brain front, image bottom is brain back, image left is brain
    # bottom, image right is brain top
    # img_2d_stack.shape (240, 240, 155); treat it as (z, y, x)
    img_2d_stack = seg_img.numpy()[0] # channel i, we only have one channel here anyways

    def plot(img_stack):
        if os.path.isdir("./out"): 
            shutil.rmtree("./out")
        os.mkdir("./out")
        for i in range(img_stack.shape[0]):
            plt.imsave(f"out/{i}.png", img_stack[i])

    z_slide = img_2d_stack  # (240, 240, 155)
    y_slide = np.swapaxes(img_2d_stack, 0, 1) # (240, 240, 155)
    x_slide = np.swapaxes(img_2d_stack, 0, 2) # (155, 240, 240)

    
if __name__ == "__main__":
    main()
    