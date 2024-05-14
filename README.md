# MDbFusion

&#x9;Thanks to reviewers' kind consideration of our manuscript, we have carefully studied the comments.

&#x9;In view of the experimental problems, we repeated the ablation experiment, and the specific results and experimental process are as follows

# Supplementary Experiment

![Qualitative Result](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/Qualitative%20Experiment.png)

![Quantitative Result](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/Quantitative%20Experiment.png)

## Experimental Configurations

&#x9;To verify the effectiveness of our Adaptive Weight Module (AWM) and the advantages of our one-stage network compared to SOTA two-stage methods, we re-conduct ablation experiment, originally shown in **Section 3.6** in the submited version, accroding to suggestions of Reviewer 06xx and Reviewer 0Exx.

&#x9;First, we set&#x20;

```math
W_{vi}^{1:N} = W_{if}^{1:N} = 0.5
```

&#x9;in **Equ.4**, which means the visible features and infrared features have the same contribution. The results are shown in the Fig. 4 and Table 3.

&#x9;To prove that motion deblurring process has a positive impact on the fusion result and prove the one-stage method outperforms two-stage method, we conduct two experiments with different processing sequences, namely Deblurring then Fusion and Fusion then Deblurring. It is noted that our network is a unified structure that cannot be strictly divided into two independent parts, so we delete fusion decoder and make rest of structue as deblurring network. Then, we choose SwinFusion \[12] as fusion network, because it has good performance in the Section 3.5.

> \[12] Jiayi Ma, Linfeng Tang, Fan Fan, Jun Huang, Xiaoguang Mei, and Yong Ma, “Swinfusion: Crossdomain long-range learning for general image fusion via&#x20;
>
> swin transformer,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 7, pp. 1200–1217, 2022.

