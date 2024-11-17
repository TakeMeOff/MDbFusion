import os
import shutil
import cv2
import numpy as np
import torch
import re
def rgb_to_ycrcb(rgb):
    rgb = rgb.numpy().transpose(1,2,0)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).transpose(2,0,1)

    return torch.from_numpy(ycrcb)

def ycrcb_to_rgb(ycrcb):
    ycrcb = ycrcb.numpy().transpose(1,2,0)
    rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB).transpose(2,0,1)
    return torch.from_numpy(rgb)



