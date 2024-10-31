# SurgPose
Official Codebase of "SurgPose:  a Dataset for Articulated Robotic Surgical Tool Pose Estimation and Tracking" 

## SAM2 for Keypoints Annotation
We provide `video_demo.py` to generate keypoints annotation files. It is adapted from the official SAM2 code.

1. Configure the environment following the official instructions https://github.com/facebookresearch/sam2.

2. Download the checkpoint (SAM2/SAM2.1) and put it in the `/checkpoints` folder.

3. Change the `ann_config` in the script to the correct information. 

4. For now, you may have to manually get (positive and/or negative) point prompts for each UV paint dot. We consider providing a GUI for easier labeling in the future.

## Stereo Matching
We provide code for stereo matching.
