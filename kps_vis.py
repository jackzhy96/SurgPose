import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse

# convert keypoints to diffent color
palette = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255),
    6: (255, 0, 255),
    7: (255, 255, 255),
    8: (128, 0, 0),
    9: (0, 128, 0),
    10: (0, 0, 128),
    11: (128, 128, 0),
    12: (0, 128, 128),
    13: (128, 0, 128),
    14: (128, 128, 128),
}

def visualize_kpts(kpt_path, frame_dir, output_dir):

    frame_list = os.listdir(frame_dir) 
    frame_list = sorted(frame_list, key=lambda x: int(x.split('.')[0][5:]))

    keypoints = yaml.load(open(kpt_path), Loader=yaml.FullLoader)

    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(frame_list):
        img = cv2.imread(os.path.join(frame_dir, frame))
        plt.figure()
        for key in keypoints[i].keys():
            if keypoints[i][key] is None:
                continue
            x, y = keypoints[i][key]
            cv2.circle(img, (int(x), int(y)), 8, palette[key], -1)

        #     plt.plot(x, y, 'ro')
        #     plt.imshow(img)
        # plt.show()
        file_name = os.path.join(output_dir, frame)
        cv2.imwrite(file_name, img)
        # print(f"Frame {i} done!")
        sys.stdout.write("\r -- Progress %s / %s" % ((i+1), len(frame_list)))
        sys.stdout.flush()
        plt.close()
    sys.stdout.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize keypoints")
    parser.add_argument("--kpt_path", type=str, help="Path to the keypoints file")
    parser.add_argument("--frame_dir", type=str, help="Path to the frames directory")
    parser.add_argument("--output_dir", type=str, help="Directory to save visualized frames")
    args = parser.parse_args()

    kpt_path = args.kpt_path
    frame_dir = args.frame_dir
    output_dir = args.output_dir

    # Call the function with the provided arguments

    visualize_kpts(kpt_path, frame_dir, output_dir)