import torchio as tio
import matplotlib.pyplot as plt


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
    
    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # print(flair_img.shape)
    # axs[0][0].imshow(flair_img.numpy()[0][10])
    # plt.savefig("test.png")

    print(flair_img.shape)
    # (1, 240, 240, 155)
    for i in range(100):
        plt.imsave(f"out/{i}.png", flair_img.numpy()[0][i])

if __name__ == "__main__":
    main()