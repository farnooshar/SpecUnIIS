This repository contains the source code for [Deep Spectral Improvement for Unsupervised Image Instance Segmentation](https://arxiv.org/abs/2401.00833)<br/>

<img src="Diagram.png">

The code has been tested with python 3.7.12 and PyTorch 1.9.1+cu111

```Shell
conda create --name specuniis
conda activate specuniis
```

## Required Data
To evaluate/train specuniis , you will need to download the required datasets.
* [Youtube-Vis 2019](https://youtube-vos.org/dataset/vis/)
* [PascalVOC2012](https://host.robots.ox.ac.uk/pascal/VOC/voc2012/) <br>
To facilitate easy testing, you can download the YouTube-VIS2019 Annotation from [here](https://drive.google.com/file/d/1SPskvTlj1tsl0uAH_ujERSCccp65WfMf/view?usp=sharing)




### Acknowledgement
This codebase is heavily borrowed from [Deep Spectral Methods for Unsupervised Localization and Segmentation](https://github.com/lukemelas/deep-spectral-segmentation). Thanks for their excellent work.
