#!/usr/bin/python3
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from joblib import Parallel, delayed

def make_sift_mask(image_path, train_path, dataset_path, sift):
    img = cv2.imread(os.path.join(train_path, image_path))
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    points = np.asarray([x.pt for x in kp])
    
    fig, ax  = plt.subplots(1,1)
    h, x, y, _ = ax.hist2d(
        points[:, 0], points[:, 1], bins=(19, 12), cmap='gray')
    ax.set_axis_off()
    fig.savefig(os.path.join(dataset_path, 'random_train_sift_mask',
                            os.path.split(
        image_path)[-1]), bbox_inches='tight', pad_inches=0)
    plt.close(fig=fig)
    return 0

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
    parser.add_argument(
        '--num_workers',
        type=int,
        help='Number of workers for parallel processing',
        default=2
    )
    args = parser.parse_args()
    sift = cv2.SIFT_create()
    os.makedirs(os.path.join(args.dataset_path, 'random_train_sift_mask'),
                exist_ok=True)
    image_paths = os.listdir(args.train_path)
    Parallel(n_jobs=args.num_workers, prefer='threads')(delayed(make_sift_mask)(image_path, args.train_path, args.dataset_path, sift) for image_path in image_paths)
    # for image_path in tqdm(image_paths):
    #     img = cv2.imread(os.path.join(args.train_path, image_path))
    #     kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    #     points = np.asarray([x.pt for x in kp])
    #     h, x, y, _ = plt.hist2d(
    #         points[:, 0], points[:, 1], bins=(19, 12), cmap='gray')
    #     fig = plt.figure()
    #     fig.axis('off')
    #     fig.savefig(os.path.join(args.dataset_path, 'random_train_sift_mask',
    #                             os.path.split(
    #         image_path)[-1]), bbox_inches='tight', pad_inches=0)
    #     plt.close(fig=fig)
