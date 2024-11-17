import torch
import torchvision.transforms
from torch.utils.data.dataset import Dataset
import os
import glob
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from augments import *
from tools import *
import matplotlib.pyplot as plt
import cv2


def tensor_to_img(tensor):
    tensor = tensor.clone().detach()  # 克隆Tensor以避免改变原始数据
    tensor = tensor.numpy().squeeze()  # 将Tensor转换为NumPy数组
    tensor = tensor.transpose(1, 2, 0)  # 调整维度
    tensor = tensor.astype(np.uint8)  # 转换为uint8类型
    img = Image.fromarray(tensor)  # 从NumPy数组创建PIL图像
    return img


def prepare_data_path(dataset_path):
    # 获取文件夹下的所有文件名字与路径
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

def save_cv2YCrCb(img, name):
    # img(C,H,W) YCrCb
    image = img.clone().cpu().numpy()
    image = np.uint8(image)
    image = np.transpose(image,(1,2,0)) #c,h,w -> h,w,c
    image = cv2.cvtColor(image,cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(name, image)

class FusionDatasets(Dataset):
    def __init__(self, vi_path, if_path, gt_path=None,
                 test=False, transform=None):
        self.trans = transforms.Compose([transforms.Resize((512, 512))])
        if not test:
            self.test = test
            self.vi_path, self.vi_names = prepare_data_path(vi_path)
            self.if_path, self.if_names = prepare_data_path(if_path)
            self.gt_path, self.gt_names = prepare_data_path(gt_path)
            self.process = self.valprocess()
            self.length = len(self.vi_path)

        else:
            self.test = test
            self.vi_path, self.vi_names = prepare_data_path(vi_path)
            self.if_path, self.if_names = prepare_data_path(if_path)
            self.process = self.valprocess()
            self.length = len(self.vi_path)

    def __len__(self):
        return self.length

    def valprocess(self):
        # TODO: add the important patch from crop, add blur
        base_transforms = Compose([

            ToTensor2()])
        return base_transforms

    def __getitem__(self, index):
        if self.test:
            vi_path = self.vi_path[index]
            if_path = self.if_path[index]
            file_name = self.vi_names[index]

            img_vi = Image.open(vi_path)
            img_if = Image.open(if_path)
            img_vi, img_if = self.process(img_vi, img_if)

            img_if = torch.cat((img_if,img_if,img_if), dim=0)
            img_vi = rgb_to_ycrcb(img_vi)
            img_if = rgb_to_ycrcb(img_if)

            return(
                img_vi,
                img_if,
                file_name)

        else:
            vi_path = self.vi_path[index]
            if_path = self.if_path[index]
            gt_path = self.gt_path[index]
            file_name = self.vi_names[index]

            img_vi = Image.open(vi_path)
            img_if = Image.open(if_path)
            img_gt = Image.open(gt_path)

            resize = transforms.Resize((512,512))
            img_vi = resize(img_vi)
            img_if = resize(img_if)
            img_gt = resize(img_gt)

            img_vi, img_if = self.process(img_vi, img_if)
            img_gt, _ = self.process(img_gt, img_gt)

            # img_if = torch.cat((img_if,img_if,img_if), dim=0)
            img_if = rgb_to_ycrcb(img_if)
            img_vi = rgb_to_ycrcb(img_vi)
            img_gt = rgb_to_ycrcb(img_gt)




            return (
                img_vi,
                img_gt,
                img_if,
                file_name)






class DeblurDataSets(Dataset):
    def __init__(self, img_path, gt_path=None, transform=False, test=False, YCrCb=False):
        if test:
            self.test = test
            self.data_path, self.data_names = prepare_data_path(img_path)
            self.length = len(self.data_path)
            self.val_process = self.valprocess()
            self.if_YCrCb = YCrCb
        else:
            self.test = test
            self.data_path, self.data_names = prepare_data_path(img_path)
            self.gt_path, self.gt_names = prepare_data_path(gt_path)
            self.transform = transform
            self.length = min(len(self.data_path), len(self.gt_path))
            self.train_preprocess = self.preprocess()
            self.if_YCrCb = YCrCb

    def __len__(self):
        return self.length

    def preprocess(self):
        # TODO: add the important patch from crop, add blur
        base_transforms = Compose([
            RandomCrop((256, 256)),
            RandomRGB(),
            RandomHorizonFlip(),
            RandomVerticalFlip(),
            RandomRotate(),
            ToTensor2()])
        return base_transforms

    def valprocess(self):
        # TODO: add the important patch from crop, add blur
        base_transforms = Compose([
            ToTensor2()])
        return base_transforms

    def __getitem__(self, index):
        if self.test:
            img_path = self.data_path[index]
            file_name = self.data_names[index]
            img = Image.open(img_path)
            img, _ = self.val_process(img, img)
            if self.if_YCrCb:
                img = rgb_to_ycrcb(img)

            return img, file_name


        else:
            img_path = self.data_path[index]
            gt_path = self.gt_path[index]
            file_name = self.data_names[index]

            img = Image.open(img_path)
            gt = Image.open(gt_path)
            '''
            if self.transform:
                # Data Augment
                trans = transforms.Compose([transforms.RandomVerticalFlip(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Resize((416, 416)),
                                            transforms.RandomRotation(degrees=90),
                                            transforms.ToTensor()])
                seed = torch.random.seed()
                torch.random.manual_seed(seed)
                img = trans(img)
                torch.random.manual_seed(seed)
                gt = trans(gt)

            else:
                totensor = torchvision.transforms.ToTensor()
                img = totensor(img)
                gt = totensor(gt)
            '''
            # resize = transforms.Resize((416, 416))
            # img = resize(img)
            # gt = resize(gt)
            img, gt = self.train_preprocess(img, gt)

            if self.if_YCrCb:
                img = rgb_to_ycrcb(img)
                gt = rgb_to_ycrcb(gt)

            '''
            img = ycrcb_to_rgb(img)
            gt = ycrcb_to_rgb(gt)
            fig1 = tensor_to_img(img)
            fig2 = tensor_to_img(gt)
            fig1.save('./fg1.png')
            fig2.save('./fg2.png')
            '''

            return (img,
                    gt,
                    file_name)
