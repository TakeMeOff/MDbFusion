import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import torch.nn as nn
import os
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def loss_calculate(img, gt):
    """
    img: [0, 1]
    gt: [0, 1]
    """
    loss_ssim = 1 - ssim(img, gt, data_range=255, size_average=True)

    eps = 1e-3
    diff = torch.add(img, -gt)
    error = torch.sqrt(diff * diff + eps * eps)
    loss_char = torch.mean(error)

    return loss_ssim, loss_char


def loss_fusion_calculate(img_vi, img_if, img_gt, img_fused):
    # 去模糊损失（只关注CrCb通道）
    loss_ssim = 1 - ssim(img_fused[:, 1:, :, :], img_gt[:, 1:, :, :], data_range=255, size_average=True)

    eps = 1e-3
    diff = torch.add(img_fused[:, 1:, :, :], -img_gt[:, 1:, :, :])
    error = torch.sqrt(diff * diff + eps * eps)
    loss_char = torch.mean(error)

    # 融合损失（只关注Y通道）
    input_max = torch.max(img_gt[:, :1, :, :], img_if[:, :1, :, :])
    loss_intensity = F.l1_loss(input_max, img_fused[:,:1,:,:])

    sobelconv = Sobelxy()
    y_grad = sobelconv(img_gt[:, :1, :, :])
    if_grad = sobelconv(img_if[:, :1, :, :])
    generate_img_grad = sobelconv(img_fused[:,:1,:,:])
    x_grad_joint = torch.max(y_grad, if_grad)
    loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

    return loss_ssim, loss_char, loss_intensity, loss_grad
