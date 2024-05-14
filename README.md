# MDbFusion

  Thanks to reviewers' kind consideration of our manuscript, we have carefully studied the comments.

  In view of the experimental suggestions, we reconduct the ablation experiment, and the experimental configuration and results analysis are as follows:

## Supplementary Experiment

![Qualitative Result](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/Qualitative%20Experiment.png)

![Quantitative Result](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/Quantitative%20Experiment.png)

### Experimental Configuration

  To verify the effectiveness of our Adaptive Weight Module (AWM) and the advantages of our one-stage network compared to SOTA two-stage methods, we re-conduct ablation experiment, originally shown in **Section 3.6** in the submited version, accroding to suggestions of **Reviewer <font color=red>06xx</font>** and **Reviewer <font color=red>0Exx</font>**.

  First, we set&#x20;

```math
W_{vi}^{1:N} = W_{if}^{1:N} = 0.5 \qquad\qquad\qquad\qquad(4)
```

in **Equ.4**, which means the visible features and infrared features have the same contribution. The results are shown in the Fig. 4 and **Table 3**.

  Second, to prove that motion deblurring process has a positive impact on the fusion result and prove the one-stage method outperforms two-stage method, we conduct two experiments with different processing sequences, namely "Deblurring then Fusion" and "Fusion then Deblurring". It is noted that our network is a unified structure that cannot be strictly divided into two independent parts, so we delete fusion decoder and make rest of structue as deblurring network. Then, we choose SwinFusion \[12] as fusion network, because it has good performance accroding to **Section 3.5**.

> \[12] Jiayi Ma, Linfeng Tang, Fan Fan, Jun Huang, Xiaoguang Mei, and Yong Ma, “Swinfusion: Crossdomain long-range learning for general image fusion via swin transformer,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 7, pp. 1200–1217, 2022.

### Results Analysis

*   As shown in **Fig. 4(c)** and **Table 3**, the result  without AWM has worse visual quality, losing  a certain amount of infrared information.
*   As shown in **Fig. 4(d)-(e)** and **Table 3**, the "Fusion then Deblurring" method fails to recover clear edges. Because the process of fusion incorporates infrared information including much noise, which makes deblurring network hard to predict potential sharp image correctly. At the same time, the "Deblurring then Fusion" method has lower indicators since such two-stage methods compress and reconstruct images twice, losing more key information than our one-stage ones.

