import json
import torch


################################################################################


TRAIN_DIR = "MICCAI_BraTS2020_TrainingData"
NUM_SAMPLES = 4
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


################################################################################


# More concise syntax to access loaded JSON object fields
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


################################################################################


def main():
    json_data = {
        "train_dir": TRAIN_DIR,
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "pin_memory": PIN_MEMORY,
    }
    with open("config.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)

if __name__ == "__main__":
    main()
    