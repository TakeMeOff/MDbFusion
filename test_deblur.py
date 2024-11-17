import torch
from model import Net
from DataReader import *
from torch.utils.data import DataLoader
from tools import *


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

if_YCrCB = True

@torch.no_grad()
def test_deblur():
    torch.cuda.set_device(0)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'))

    weight_path = 'Your Weight Path'
    image_save_path = './Results/deblur'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, mode=0o777, exist_ok=True)

    model = Net(input_channels=3, output_channels=3)
    model.load_state_dict(torch.load(weight_path), strict=False)
    print("Loading Path Successfully!")

    model = model.to(device)
    model.eval()

    test_datasets = DeblurDataSets(img_path='Your Blurry Image Path', test=True, YCrCb=if_YCrCB)
    test_loader = DataLoader(test_datasets,
                             batch_size=1,
                             num_workers=8,
                             shuffle=False,
                             drop_last=False)

    model.eval()
    for it, (img, name) in enumerate(test_loader):


        img = img.to(device)
        result = model(img)
        result = result.cpu()



        for k in range(len(name)):
            image = result[k, :, :, :]
            if if_YCrCB:
                image = ycrcb_to_rgb(image)
            image = tensor_to_img(image.clamp_(0, 255))
            save_path = os.path.join(image_save_path, name[k])
            image.save(save_path)
            print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == '__main__':
    test_deblur()
