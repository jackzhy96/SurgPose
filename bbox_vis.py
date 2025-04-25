import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

def visualize_bbox(bbox_path, frame_dir, output_dir):
    frame_list = os.listdir(frame_dir) 
    frame_list = sorted(frame_list, key=lambda x: int(x.split('.')[0][5:]))

    # load .json bbox data
    with open(bbox_path, 'r') as f:
        bbox_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for idx, frame in enumerate(frame_list):
        img = cv2.imread(os.path.join(frame_dir, frame))
        for key in bbox_data[str(idx)].keys():
            if bbox_data[str(idx)][key] is None:
                continue
            x, y, w, h = bbox_data[str(idx)][key]
            # Draw the bounding box on the image
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        file_name = os.path.join(output_dir, frame)
        cv2.imwrite(file_name, img)
        print(f"Frame {idx} done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize bounding boxes on frames")
    parser.add_argument("--bbox_path", type=str, help="Path to the bounding box json file")
    parser.add_argument("--frame_dir", type=str, help="Path to the frames directory")
    parser.add_argument("--output_dir", type=str, help="Directory to save visualized frames")
    args = parser.parse_args()

    bbox_path = args.bbox_path
    frame_dir = args.frame_dir
    output_dir = args.output_dir

    visualize_bbox(bbox_path, frame_dir, output_dir)
