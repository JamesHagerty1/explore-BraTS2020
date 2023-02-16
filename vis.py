import torchio as tio
import matplotlib.pyplot as plt
import numpy as np


def main():
    flair_path = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
    t1_path = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii'
    t1ce_path = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii'
    t2_path = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii'
    seg_path = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'
    
    flair_img = tio.ScalarImage(flair_path)
    t1_img = tio.ScalarImage(t1_path)
    t1ce_img = tio.ScalarImage(t1ce_path)
    t2_img = tio.ScalarImage(t2_path)
    seg_img = tio.LabelMap(seg_path)

    # img_2d_stack default is sideways brain view where:
    # image top is brain front, image bottom is brain back, image left is brain
    # bottom, image right is brain top
    # img_2d_stack.shape (240, 240, 155), arbitrarily treat it as (z, y, x)
    img_2d_stack = flair_img.numpy()[0] # batch i

    top_brain_stack = np.swapaxes(img_2d_stack, 0, 2) # (x, y, z); (155, 240, 240)
    for i in range(40, 60):
        plt.imsave(f"out/{i}.png", top_brain_stack[i])


    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # print(flair_img.shape)
    # axs[0][0].imshow( numpy arr )
    # plt.savefig("test.png")

    # also try plt for 3d data!

if __name__ == "__main__":
    main()