## Toy Demo of "Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration"

### Install 
+ Environment
    ```
    conda create -n CMR python=3.6
    conda activate CMR
    ```
+ Please follow [official suggestions](https://pytorch.org/) to install pytorch and torchvision. We use pytorch=1.5.0, torchvision=0.6.0
+ Requirements
    ```
    pip install -r requirements.txt
    ```
+ [MPI-IS Mesh](https://github.com/MPI-IS/mesh): We suggest to install this library from the sourcem [here](https://drive.google.com/file/d/1Lfz2Tjo8opjCZbcmyIYhqQcGwhasIsvp/view?usp=sharing) or [here](https://drive.google.com/file/d/1BOzIuMG8KJ92ZdzdV_GHYrIeWd4W2Vhk/view?usp=sharing), and place it at `out/FreiHAND/cmr_pg/checkpoints/cmr_pg_res18_freihand.pt` or `out/FreiHAND/cmr_sp/checkpoints/cmr_sp_res50_freihand.pt`

### Run a demo
```
./demo.sh
```
The prediction results will be saved in `out/FreiHAND/cmr_pg/demo` 

### Evaluation on FreiHAND
#### Dataset
Please download FreiHAND dataset from [here](https://lmb.informatik.uni-freiburg.de/projects/freihand/), and create a soft link in `data`, i.e., `data/FreiHAND`.
#### Run
```
./eval_freihand.sh
```
+ JSON file will be saved as `out/FreiHAND/cmr_pg/cmr_pg.josn`. You can submmit this file to the [official server](https://competitions.codalab.org/competitions/21238) for evaluation. Note that because this is a online competition, we did not provide our best model.
+ If you want to save prediction results like above demo, you would want to uncomment Line 86 in `run.py`. The prediction results will be saved in `out/FreiHAND/cmr_pg/eval`.

## Explaination of output
+ In an JPEG file (e.g., 000_plot.jpg), we show silhouette, 2D pose, projection of mesh, camera-space mesh and pose
+ As for camera-space information, we use a red rectangle to indicate the camera position, or the image plane.
    + The unit is meter.
    + Please refer to our paper for X-, Y-, and Z-axes.
+ If you run the demo, you can also obtain a PLY file (e.g., 000_mesh.ply). 
    + This file is a 3D model of the hand.
    + You can open it with corresponding software (e.g., Preview in Mac).
    + Here, you can get more 3D details through rotation and zoom in.
