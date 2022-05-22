# Complement data

We develop a synthetic dataset with 1520 poses and 216 viewpoints (in which 71 are released for research purpose), both of which are uniformly distributed in their respective spaces. Because of this superior property, it can serve as a good complement during training. Please refer to [MobRecon Supplement](https://arxiv.org/abs/2112.02753) for details.

You can download files from [here](https://drive.google.com/drive/folders/1V3Ioy3H1vGPG4mURsCon9TE7j5eGFanN) and unzip it.

If you use this dataset in your research, please cite:

```
@inproceedings{bib:MobRecon,
  title={MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image},
  author={Chen, Xingyu and Liu, Yufeng and Dong Yajiao and Zhang, Xiong and Ma, Chongyang and Xiong, Yanmin and Zhang, Yuan and Guo, Xiaoyan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
```

BUG REPORT: the synthetic global pose for base pose 2 is unchanged as the background rotating. This bug has ignorable effect in terms of model performance, and we plan to solve this issue in the future. 

