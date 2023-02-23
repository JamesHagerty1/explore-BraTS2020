import json
import os
import torch


################################################################################


TRAIN_DIR = "MICCAI_BraTS2020_TrainingData"
NUM_LABELS = -1

NUM_SAMPLES = 4
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

ENC_CONV_CHANNELS = ((1, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512))
DEC_UPCONV_CHANNELS = ((512, 512), (256, 256), (128, 128))
# excluding decoder conv for inference, which is inferred by NUM_LABELS
DEC_CONV_CHANNELS = ((256+512, 256, 256), (128+256, 128, 128), (64+128, 64, 64))
CONV_KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 2
UPCONV_KERNEL_SIZE = 2


################################################################################


# More concise syntax to access loaded JSON object fields
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


################################################################################


def main():
    if os.path.exists("./config.json"):
        os.remove("./config.json")
    json_data = {
        "train_dir": TRAIN_DIR,
        "num_labels": NUM_LABELS,
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "pin_memory": PIN_MEMORY,
        "enc_conv_channels": ENC_CONV_CHANNELS,
        "dec_upconv_channels": DEC_UPCONV_CHANNELS,
        "dec_conv_channels": DEC_CONV_CHANNELS,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "pool_kernel_size": POOL_KERNEL_SIZE,
        "upconv_kernel_size": UPCONV_KERNEL_SIZE,
    }
    with open("./config.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)

if __name__ == "__main__":
    main()
