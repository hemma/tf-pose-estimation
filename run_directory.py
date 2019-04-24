import argparse
import ast
import csv
import glob
import logging
import os
import time

import cv2
import tensorflow as tf

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator, Human
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from typing import List


def pack_humans(humans: List[Human]):
    p_humans = []
    for human in humans:
        body_parts = [-1 for x in range(18 * 3)]
        for part in human.body_parts.values():
            body_parts[part.part_idx * 3] = round(part.x, 3)
            body_parts[part.part_idx * 3 + 1] = round(part.y, 3)
            body_parts[part.part_idx * 3 + 2] = round(part.score, 3)

        if len(human.body_parts.values()) > 4:
            p_humans.append(body_parts)

    return p_humans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--out', type=str, default='./pose-coordinates.csv')
    parser.add_argument('--folder', type=str, nargs='+', default='./images/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    args = parser.parse_args()


    w, h = model_wh(args.resolution)

    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h),
                            tf_config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368),
                            tf_config=tf.ConfigProto(gpu_options=gpu_options))

    files_grabbed = sum([glob.glob(os.path.join(folder, '*.jpg')) for folder in args.folder], [])

    packed_humans = {}
    for i, file in enumerate(files_grabbed):
        # estimate human poses from a single image !
        image = common.read_imgfile(file, None, None)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        elapsed = time.time() - t

        logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey(5)

        if len(humans) > 1:
            print('Multiple humans in %s' % file)

        packed = pack_humans(humans)
        if len(packed) >= 1:
            packed_humans[file] = packed
        else:
            print('No humans found in %s' % file)

    with open(args.out, 'w') as file:
        writer = csv.writer(file)

        for key in packed_humans.keys():
            print(key)
            writer.writerows(packed_humans[key])
