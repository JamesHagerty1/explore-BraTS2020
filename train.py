import json
import os
from torch.utils.data import DataLoader
from gen_config import AttrDict
from dataset import BraTS2020
from model import UNet3D


################################################################################


def main():
    with open("./config.json") as json_file:
        json_data = json_file.read()
    config = json.loads(json_data) 
    c = AttrDict(config) # config, concise JSON object access syntax

    dataset = BraTS2020(c)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=c.batch_size, 
        pin_memory=c.pin_memory, num_workers=os.cpu_count())
    
    model = UNet3D(c)

    for i, (x, y) in enumerate(dataloader):
        y_hat = model(x)
        break

if __name__ == "__main__":
    main()
