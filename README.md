# SurgPose: a Dataset for Articulated Robotic Surgical Tool Pose Estimation and Tracking
Accepted by ICRA 2025! See you in Atlanta! 

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)



## SAM2 for Keypoints Annotation
We provide `kps_annotation_sam2.py` to generate keypoints annotation files. It is adapted from the official SAM2 code.

1. Configure the environment following the official instructions https://github.com/facebookresearch/sam2.

2. Download the checkpoint (SAM2/SAM2.1) and put it in the `/checkpoints` folder.

3. Change the `ann_config` in the script to the correct information. 

4. For now, you may have to manually get (positive and/or negative) point prompts for each UV paint dot. We consider providing a GUI for easier labeling in the future.

## Stereo Depth Estimation
We provide example code for stereo matching based on [RAFT](https://github.com/princeton-vl/RAFT). 

1. <b>Stereo Camera Calibration:</b> SurgPose includes stereo camera parameters `StereoCalibrationDVRK.ini` deriving from [MATLAB 2024b Stereo Camera Calibration](https://www.mathworks.com/help/vision/ug/using-the-stereo-camera-calibrator-app.html) Application. In `StereoCalibrationDVRK.ini`, k0, k1, k4 refer radial distortion coeffients and k2, k3 infer tangential distortion coeffients.

2. <b>Stereo Matching:</b> Run `python depth_estimator.py -d [path to data, e.g. /SurgPose/000000]`. This script is modified from [Deform3DGS](). Note that the image size of SurgPose is 1400x986. In this example script, all images will be resized to 696x488.

## Trajectory Generation

1. Define the workspace for PSM1 and/or PSM3 (PSM2).

2. Randomly sample N points in the workspace.

3. Generate smooth and periodic end-effector trajectories passing the N points, based on the Curve Fitting Toolbox and Robotic System Toolbox in MATLAB 2024b. Please be aware that this trajectory only have translation, WITHOUT rotation.

4. For articulations, we use periodic functions (sine and cosine) to define the trajectory of each degree of freedom (shaft rotation, wrist joint, gripper rotation, and gripper open angle). You may need to modify parameters of these functions to fit your system and applications.

5. After above steps, you are good to run code `trajectory_generator.m` in MATLAB.

6. Use `` to parse the generated PSM trajectory to make it can be used in Python.

## Robot Movement and Data Collection

SurgPose was collected in the [Robotics and Control Laboratory](https://rcl.ece.ubc.ca/) @ UBC. We used da Vinci IS1200 system with the [da Vinci Research Kit](https://github.com/jhu-dvrk) (dVRK). We collected more data for evaluation in the [Laboratory for Computational Sensing and Robotics](https://lcsr.jhu.edu/) @ JHU and used da Vinci SI system with dVRK. 

### Setup in RCL @ UBC
The dVRK version is 2.1. Run `python data_collection.py`

### Setup in LCSR @ JHU

## Annotations

### Keypoints

### Forward Kinematics & Joint States from dVRK

### Segmentation Mask

## Material List

## Related Works and Benchmarks 

Here we list a few baselines that can be used for Surgical Instrument Pose Estimation. If you have other methods for SurgPose, please reach out and we will add them here.

1. [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut): There is an example tutorial @[RandyMoore](https://github.com/rwjmoore): [SurgPose performance with deeplabcut](https://github.com/rwjmoore/surgPose_deeplabcut)

2. [YOLO v8](https://docs.ultralytics.com/models/yolov8/): The annotation need to be reformatted to YOLO format.

3. [ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://github.com/ViTAE-Transformer/ViTPose) The annotation need to be reformated to COCO format.

## Acknowledgement
We sincerely appreciate [RAFT](), [Segment Anything 2](), [Deform3DGS](). Many thanks to these fantastic works and their open-sourse contribution!

## Citation
If you feel SurgPose or this codebase is useful, please consider cite this paper:
```
```

## Contact
If you have any issues, feel free to reach out zijianwu@ece.ubc.ca

