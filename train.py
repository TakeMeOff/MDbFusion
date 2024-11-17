import torch
import os
from torch.utils.data import DataLoader
from model import Net, FusionNet
from DataReader import DeblurDataSets, FusionDatasets
import numpy as np
import random
from Loss import loss_calculate, loss_fusion_calculate
import logging
import matplotlib.pyplot as plt
from DataReader import tensor_to_img
import torchvision.transforms as transforms


def plot_cure(data, x_label, y_label, title, file_name):
    fig = plt.figure()
    x_axis = range(len(data))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, data, label=title)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(file_name)


def train_deblur():
    # Set Device
    torch.cuda.set_device(0)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'))
    print("The training device is:{}".format(device))
    logging.info("The training device is:{}".format(device))

    # Relative Parameters
    epochs = 2000
    lr_initial = 4e-4
    batch_size = 4
    num_workers = 8
    weight_decay = 1e-4
    pre_epochs = 0
    is_continue = False
    if_YCrCb = False

    # Set path
    weight_path = './Weights/deblur_M3FD_Motion'
    if not os.path.exists('./Weights'):
        os.makedirs('./Weights')
    continue_path = 'Your Last Weight Path'

    # Loading Training Sets
    train_dataset = DeblurDataSets(img_path='Your Motion Images Path', gt_path='Your GT Path',
                                   transform=True, YCrCb=if_YCrCb)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    print("Loading Training Sets Successfully!")
    logging.info("Loading Training Sets Successfully!")

    # Model
    model = Net(input_channels=3, output_channels=3)
    model = model.to(device)
    if is_continue:
        model.load_state_dict(torch.load(continue_path))
        print("Loading Weights Successfully!")
        logging.info("Loading Weights Successfully!")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_initial, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    # Initial
    times = 0
    times_ssim = 0
    times_char = 0

    fig_ssim = []
    fig_char = []

    # Training loop
    for epoch in range(epochs):
        model = model.to(device)
        model.train()
        print("Epoch:{}/{}  Training".format(epoch + 1 + pre_epochs, epochs))
        logging.info("Epoch:{}/{}  Training".format(epoch + 1 + pre_epochs, epochs))
        for it, (img, gt, name) in enumerate(train_loader):
            img = img.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            result = model(img.to(device))
            loss_ssim, loss_char = loss_calculate(result, gt)
            loss = loss_ssim + loss_char

            loss.backward()
            optimizer.step()

            # Save the loss
            times_ssim += loss_ssim.item()
            times_char += loss_char.item()

            times += 1

            if (times + 1) % 100 == 0:
                print("epoch: {}, loss_ssim: {:.4f}, loss_char: {:.4f}, lr: {:.5f}".format(epoch + 1 + pre_epochs,
                                                                                           times_ssim / 100,
                                                                                           times_char / 100,
                                                                                           scheduler.get_last_lr()[
                                                                                               0]))
                logging.info(
                    "epoch: {}, loss_ssim: {:.4f}, loss_char: {:.4f}, lr: {:.5f}".format(epoch + 1 + pre_epochs,
                                                                                         times_ssim / 100,
                                                                                         times_char / 100,
                                                                                         scheduler.get_last_lr()[
                                                                                             0]))
                fig_ssim.append(times_ssim / 100)
                fig_char.append(times_char / 100)
                times_ssim = 0
                times_char = 0

        scheduler.step()
        if (epoch + 1) % 100 == 0:
            torch.save(model.cpu().state_dict(), f'{weight_path}_epoch_{epoch + pre_epochs}.pth')
            print("Save {}th Path".format(epoch + 1 + pre_epochs))
            logging.info("Save {}th Path".format(epoch + 1 + pre_epochs))

    torch.save(model.cpu().state_dict(), f'{weight_path}_epoch_{epochs + pre_epochs}.pth')
    print("Deblur Training Successfully!")
    logging.info("Deblur Training Successfully!")

    plot_cure(data=fig_ssim, x_label='batches', y_label='Loss-SSIM', title='Loss-SSIM', file_name='./Loss SSIM.png')
    plot_cure(data=fig_char, x_label='batches', y_label='Loss-char', title='Loss-char', file_name='./Loss char.png')


def train_fusion():
    # Set Device
    torch.cuda.set_device(0)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'))
    print("The training device is:{}".format(device))
    logging.info("The training device is:{}".format(device))

    # Relative Parameters
    epochs = 200
    lr_initial = 1e-4
    batch_size = 4
    num_workers = 8
    weight_decay = 1e-4
    is_continue = False
    if_YCrCb = True

    # Set path
    if not os.path.exists('./Weights'):
        os.makedirs('./Weights')
    Deblur_path = './Weights/deblur_M3FD_Motion_epoch_2000.pth'
    Fusion_path = './Weights/fusion_M3FD_Motion_200.pth'

    # Loading Training Sets
    fusion_dataset = FusionDatasets(vi_path='/mnt/data0/MDFusionV2/Data/M3FD_Motion/train/img_vi',
                                    if_path='/mnt/data0/MDFusionV2/Data/M3FD_Motion/train/img_if',
                                    gt_path='/mnt/data0/MDFusionV2/Data/M3FD_Motion/train/img_gt',
                                    test=False)

    train_loader = DataLoader(fusion_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    print("Loading Training Sets Successfully!")
    logging.info("Loading Training Sets Successfully!")

    # Model
    model = FusionNet(input_channels=3, output_channels=3)
    model = model.to(device)

    if is_continue:
        model.load_state_dict(torch.load(Deblur_path), strict=False)
        print("Loading Deblur Weight Successfully!")
        logging.info("Loading Deblur Weight Successfully!")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_initial, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr_initial * 0.1)

    # Initial
    times = 0
    times_ssim = 0
    times_char = 0
    times_intensity = 0
    times_grad = 0
    fig_ssim = []
    fig_char = []
    fig_intensity = []
    fig_grad = []

    # Training loop
    for epoch in range(epochs):
        model = model.to(device)
        model.train()
        print("Epoch:{}/{}  Training".format(epoch + 1, epochs))
        logging.info("Epoch:{}/{}  Training".format(epoch + 1, epochs))
        for it, (img_vi, gt, img_if, name) in enumerate(train_loader):


            img_vi = img_vi.to(device)
            img_if = img_if.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            result = model(img_vi, img_if)
            loss_ssim, loss_char, loss_intensity, loss_grad = loss_fusion_calculate(img_vi, img_if, gt, result)
            loss = loss_ssim + loss_char + loss_intensity * 2 + loss_grad * 3

            loss.backward()
            optimizer.step()

            times_ssim += loss_ssim.item()
            times_char += loss_char.item()
            times_intensity += loss_intensity.item()
            times_grad += loss_grad.item()
            times += 1

            if (times + 1) % 10 == 0:
                print(
                    "epoch: {}, loss_ssim: {:.4f}, loss_char: {:.4f}, loss_in: {:.4f}, loss_grad: {:.4f}, lr: {:.5f}".format(
                        epoch + 1,
                        times_ssim / 10,
                        times_char / 10,
                        times_intensity / 10,
                        times_grad / 10,
                        scheduler.get_last_lr()[0]))
                logging.info(
                    "epoch: {}, loss_ssim: {:.4f}, loss_char: {:.4f}, loss_in: {:.4f}, loss_grad: {:.4f}, lr: {:.5f}".format(
                        epoch + 1,
                        times_ssim / 10,
                        times_char / 10,
                        times_intensity / 10,
                        times_grad / 10,
                        scheduler.get_last_lr()[0]))
                fig_ssim.append(times_ssim / 10)
                fig_char.append(times_char / 10)
                fig_intensity.append(times_intensity / 10)
                fig_grad.append(times_grad / 10)
                times_ssim = 0
                times_char = 0
                times_intensity = 0
                times_grad = 0

        scheduler.step()
    torch.save(model.cpu().state_dict(), Fusion_path)
    print("Deblur Training Successfully!")
    logging.info("Deblur Training Successfully!")
    plot_cure(data=fig_ssim, x_label='batches', y_label='Loss-SSIM', title='Loss-SSIM', file_name='./Fusion SSIM.png')
    plot_cure(data=fig_char, x_label='batches', y_label='Loss-char', title='Loss-char', file_name='./Fusion char.png')
    plot_cure(data=fig_intensity, x_label='batches', y_label='Loss-ntensit', title='Loss-ntensit', file_name='./Fusion intensity.png')
    plot_cure(data=fig_grad, x_label='batches', y_label='Loss-grad', title='Loss-grad', file_name='./Fusion grad.png')



if __name__ == '__main__':
    logging.basicConfig(filename='./logging.txt',
                        filemode="w",
                        format="【%(asctime)s】：%(message)s",
                        level=logging.INFO)
    train_fusion()
