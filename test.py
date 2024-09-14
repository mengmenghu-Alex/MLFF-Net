import numpy as np
import torchvision
from torchvision import transforms
import argparse
from Net.model import *
from Utils.dataloader import myDataSet
import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
__all__ = [
    "test",
    "setup",
    "testing",
]

@torch.no_grad()
def test(config, test_dataloader, test_model):
    test_model.eval()
    for i, (img, _, name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(config.device)
            out = test_model(img)
            torchvision.utils.save_image(out, config.enhance_images_path + name[0])

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    test_model = torch.load(config.weight_path).to(config.device)
    test_dataset = myDataSet(config.test_images_path,None,transforms.ToTensor(), False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = config.batch_size,shuffle = False)
    print("Test Dataset Reading Completed.")
    return test_dataloader, test_model

def testing(config):
    ds_test, model = setup(config)
    test(config, ds_test, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default="./Weights/UIEB.pth")
    parser.add_argument('--label_images_path', type=str, default="./Data/test/label/")
    parser.add_argument('--test_images_path', type=str, default="./Data/test/input/")
    parser.add_argument('--enhance_images_path', type=str, default='./Data/test/output/')
    parser.add_argument('--batch_size', type=int, default=1)
    
    config = parser.parse_args()
    testing(config)