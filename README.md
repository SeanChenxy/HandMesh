
# Hand Mesh Reconstruction


## Introduction
This repo is the PyTorch implementation of hand mesh reconstruction described in [CMR](https://arxiv.org/abs/2103.02845) and [MobRecon](https://arxiv.org/abs/2112.02753).

## Update
+ 2021-12.7, Add MobRecon demo.
+ 2021-6-10, Add Human3.6M dataset.
+ 2021-5-20, Add CMR-G model.

## Features
- [x] SpiralNet++
- [x] Sub-pose aggregation
- [x] Adaptive 2D-1D registration for mesh-image alignment
- [x] DenseStack for 2D encoding
- [x] Feature lifting with MapReg and PVL
- [x] DSConv as an efficient mesh operator
- [ ] MobRecon training with consistency learning and complement data

## Install 
+ Environment
    ```
    conda create -n handmesh python=3.6
    conda activate handmesh
    ```
+ Please follow [official suggestions](https://pytorch.org/) to install pytorch and torchvision. We use pytorch=1.7.1, torchvision=0.8.2
+ Requirements
    ```
    pip install -r requirements.txt
    ```
  If you have difficulty in installing `torch_sparse` etc., please use `whl` file from [here](https://pytorch-geometric.com/whl/).
+ [MPI-IS Mesh](https://github.com/MPI-IS/mesh): We suggest to install this library from the source 

+ Download the files you need from [Google drive](https://drive.google.com/drive/folders/1MIE0Jo01blG6RWo2trQbXlQ92tMOaLx_?usp=sharing).

## Run a demo
+ Prepare pre-trained models as
  ```
  out/Human36M/cmr_g/checkpoints/cmr_pg_res18_human36m.pt
  out/FreiHAND/cmr_g/checkpoints/cmr_g_res18_moredata.pt
  out/FreiHAND/cmr_sg/checkpoints/cmr_sg_res18_freihand.pt
  out/FreiHAND/cmr_pg/checkpoints/cmr_pg_res18_freihand.pt  
  out/FreiHAND/mobrecon/checkpoints/mobrecon_densestack_dsconv.pt  
  ``` 
+ Run
  ```
  ./scripts/demo_cmr.sh
  ./scripts/demo_mobrecon.sh
  ```
  The prediction results will be saved in output directory, e.g., `out/FreiHAND/mobrecon/demo`.

+  Explaination of the output

    <p align="middle">  
    <img src="./images/2299_plot.jpg">  
    </p> 

    + In an JPEG file (e.g., 000_plot.jpg), we show silhouette, 2D pose, projection of mesh, camera-space mesh and pose
    + As for camera-space information, we use a red rectangle to indicate the camera position, or the image plane. The unit is meter.
    + If you run the demo, you can also obtain a PLY file (e.g., 000_mesh.ply). 
        + This file is a 3D model of the hand.
        + You can open it with corresponding software (e.g., Preview in Mac).
        + Here, you can get more 3D details through rotation and zoom in.

## Dataset
#### FreiHAND
+ Please download FreiHAND dataset from [this link](https://lmb.informatik.uni-freiburg.de/projects/freihand/), and create a soft link in `data`, i.e., `data/FreiHAND`.
+ Download mesh GT file `freihand_train_mesh.zip`, and unzip it under `data/FreiHAND/training`
#### Human3.6M
+ The official data is now not avaliable. Please follow [I2L repo](https://github.com/mks0601/I2L-MeshNet_RELEASE) to download it.
+ Download silhouette GT file `h36m_mask.zip`, and unzip it under `data/Human36M`.
#### Data dir
```  
${ROOT}  
|-- data  
|   |-- FreiHAND
|   |   |-- training
|   |   |   |-- rgb
|   |   |   |-- mask
|   |   |   |-- mesh
|   |   |-- evaluation
|   |   |   |-- rgb
|   |   |-- evaluation_K.json
|   |   |-- evaluation_scals.json
|   |   |-- training_K.json
|   |   |-- training_mano.json
|   |   |-- training_xyz.json
|   |-- Human3.6M
|   |   |-- images
|   |   |-- mask
|   |   |-- annotations
```  

## Evaluation
#### FreiHAND
```
./scripts/eval_cmr_freihand.sh
./scripts/eval_mobrecon_freihand.sh
```
+ JSON file will be saved as `out/FreiHAND/cmr_sg/cmr_sg.josn`. You can submmit this file to the [official server](https://competitions.codalab.org/competitions/21238) for evaluation.

#### Human3.6M
```
./scripts/eval_cmr_human36m.sh
```
#### Performance on PA-MPJPE (mm)
We re-produce the following results after code re-organization.

|  Model / Dataset   | FreiHAND  | Human3.6M (w/o COCO) |
|  :----:  | :----:  |:----:  |
| CMR-G-ResNet18   | 7.6 | - |
| CMR-SG-ResNet18  | 7.5 | - |
| CMR-PG-ResNet18  | 7.5 | 50.0 |
| MobRecon-DenseStack  | 6.9 | - |

## Training
```
./scripts/train_cmr_freihand.sh
./scripts/train_cmr_human36m.sh
```
## Reference
```tex
@inproceedings{bib:CMR,
  title={Camera-Space Hand Mesh Recovery via Semantic Aggregationand Adaptive 2D-1D Registration},
  author={Chen, Xingyu and Liu, Yufeng and Ma, Chongyang and Chang, Jianlong and Wang, Huayan and Chen, Tian and Guo, Xiaoyan and Wan, Pengfei and Zheng, Wen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
@article{bib:MobRecon,
  title={MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image},
  author={Chen, Xingyu and Liu, Yufeng and Dong Yajiao and Zhang, Xiong and Ma, Chongyang and Xiong, Yanmin and Zhang, Yuan and Guo, Xiaoyan},
  journal={arXiv:2112.02753},
  year={2021}
}
}
```

## Acknowledgement
Our implementation of SpiralConv is based on [spiralnet_plus](https://github.com/sw-gong/spiralnet_plus?utm_source=catalyzex.com).
