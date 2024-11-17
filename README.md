# MDbFusion

This is official Pytorch implementation of "[Mdbfusion: A Visible And Infrared Image Fusion Framework Capable For Motion Deblurring](https://ieeexplore.ieee.org/document/10647563)"

## Network Architecture

![image](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/fig1.png)

The overall structure of image fusion framework capable for motion deblurring (MDbFusion)

## To train

The processes of training MDbFusion are divided into two stages:

1.  First, Train the encoder-decoder structure with GoPro Datasets. Users can download training dataset from below website:

    [Publications Datasets CV | Seungjun Nah](https://seungjunnah.github.io/Datasets/gopro.html).&#x20;

    Then, run `train_derblur()` in `train.py`.


2.  &#x20;Second, train the whole network struture with simulated datasets based on LLVIP datasets. The LLVIP datasets can be download from below website:

    [LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](https://bupt-ai-cz.github.io/LLVIP/)

    Then, run `train_fusion()` in `train.py`.

## To Test

Run `test_fusion.py` to predict deblurring fused images.

## Fusion Results

### Without Pre-processing:

![image](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/fig2.png)

### With Pre-deblurring:

![image](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/fig3.png)

#### If this work is helpful to you, please cite it as:

    @INPROCEEDINGS{10647563,
      author={Chen, Jun and Yu, Wei and Tian, Xin and Huang, Jun and Ma, Jiayi},
      booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
      title={Mdbfusion: A Visible And Infrared Image Fusion Framework Capable For Motion Deblurring}, 
      year={2024},
      pages={1019-1025}}

