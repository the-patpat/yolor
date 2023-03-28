#!/usr/bin/python3
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_path',
        type=str,
        help='Path to directory containing train images'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to dataset')
    args = parser.parse_args()
    sift = cv2.SIFT_create()
    os.makedirs(os.path.join(args.dataset_path, 'random_train_sift_mask'),
                exist_ok=True)
    image_paths = os.listdir(args.train_path)
    for image_path in tqdm(image_paths):
        img = cv2.imread(os.path.join(args.train_path, image_path))
        kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        points = np.asarray([x.pt for x in kp])
        h, x, y, _ = plt.hist2d(
            points[:, 0], points[:, 1], bins=(19, 12), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.dataset_path, 'random_train_sift_mask',
                                os.path.split(
            image_path)[-1]), bbox_inches='tight', pad_inches=0)
