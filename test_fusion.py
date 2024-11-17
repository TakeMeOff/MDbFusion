import torch
import os
from model import FusionNet, Net
from DataReader import *
from torch.utils.data import DataLoader
import numpy as np
import cv2


@torch.no_grad()
def test_fusion():
    torch.cuda.set_device(0)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'))

    weight_path = 'Your Weight Path'
    image_save_path = './Results/fusion'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, mode=0o777, exist_ok=True)

    model = FusionNet(input_channels=3, output_channels=3)

    # model.load_state_dict(torch.load(Deblur_path), strict=False)
    print("Loading weight successfully")
    model.load_state_dict(torch.load(weight_path))


    print("Loading Path Successfully!")

    model = model.to(device)
    model.eval()

    test_datasets = FusionDatasets(vi_path='Visible Path', if_path='Infrared Path', test=True)
    test_loader = DataLoader(test_datasets,
                             batch_size=1,
                             num_workers=8,
                             shuffle=False,
                             drop_last=False)

    for it, (img_vi, img_if, name) in enumerate(test_loader):
        img_vi = img_vi.to(device)
        img_if = img_if.to(device)
        result = model(img_vi, img_if)

        for k in range(len(name)):
            img = result[k, :, :, :].cpu()
            image = ycrcb_to_rgb(img)
            image = tensor_to_img(image.clamp_(0, 255))
            save_path = os.path.join(image_save_path, name[k])
            image.save(save_path)
            print('Fusion {0} Successfully!'.format(save_path))

if __name__ == '__main__':
    test_fusion()