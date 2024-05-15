# MDbFusion

We have careflly studied the reviewers' comments, thanks to their kind consideration of our manuscript.

In response to the experiment suggestions, we have reconducted the ablation experiments, and the experiment configurations as well as results analysis are presented as follows:

## Supplementary Experiments

<div align="center">
  
  ![Qualitative Result](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/Qualitative%20Experiment.png)

</div>

<div align="center">
  
  ![Quantitative Result](https://github.com/TakeMeOff/MDbFusion/blob/main/fig/Quantitative%20Experiment.png)

</div>

### Experiment Configurations

To verify the effectiveness of our Adaptive Weight Module (AWM) and the advantages of our one-stage network compared to state-of-the-art SOTA two-stage methods, we have reconducted ablation experiments, originally presented in **Section 3.6** in the submited version, according to suggestions of **Reviewer 06xx** and **Reviewer 0Exx**.

First, we set&#x20;

```math
W_{vi}^{1:N} = W_{if}^{1:N} = 0.5 \qquad\qquad\qquad\qquad(4)
```

in **Equ.4**, which means the visible features and infrared features have the same contribution. The results are shown in the Fig. 4 and **Table 3**.

Second, to prove that motion deblurring process has a positive impact on the fusion result and that the one-stage method outperforms the two-stage method, we conduct two experiments with different processing sequences:  "Deblurring then Fusion" and "Fusion then Deblurring". It is important to note that our network has a unified structure that cannot be strictly divided into two independent parts. Therefore, to simulate the two-stage process, we modify our network by removing the fusion decoder and utilize the remaining structure as a deblurring network. Subsequently, we choose SwinFusion \[12] as the fusion network, as it has shown good performance according to **Section 3.5**.&#x20;

> \[12] Jiayi Ma, Linfeng Tang, Fan Fan, Jun Huang, Xiaoguang Mei, and Yong Ma, “Swinfusion: Crossdomain long-range learning for general image fusion via swin transformer,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 7, pp. 1200–1217, 2022.

### Results Analysis

*   As shown in **Fig. 4(c)** and **Table 3**, the result without AWM exhibits inferior visual quality, resulting in a loss of certain amount of infrared information.
*   As shown in **Fig. 4(d)-(e)** and **Table 3**, the "Fusion then Deblurring" method fails to recover clear edges. This is because the fusion process incorporates infrared information that contains significant noise, which makes deblurring network hard to predict potential sharp image correctly. In contrast, the "Deblurring then Fusion" method achieves lower indicators since such two-stage methods compress and reconstruct images twice, resulting in losing more crucial information compared to our one-stage approach.

