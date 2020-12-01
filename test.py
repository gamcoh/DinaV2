import argparse
import configparser
import os
import sys
from ast import literal_eval
from collections import deque
from typing import Deque

import cv2

from lib.model_res import Resnet3DBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    dest="config",
                    help="Configuration file used to run the script",
                    required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
nb_frames: int = config.getint('general', 'nb_frames')
nb_classes: int = config.getint('general', 'nb_classes')
model_name: str = config.get('path', 'model_name')
data_root: str = config.get('path', 'data_root')
data_model: str = config.get('path', 'data_model')
target_size: tuple  = literal_eval(config.get('general', 'target_size'))
path_weights: str = config.get('path', 'path_weights')

cap = cv2.VideoCapture(0)

# Loading the model
path_model = os.path.join(data_root, data_model, model_name)
inp_shape: tuple = (nb_frames,) + target_size + (3,)
net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes)
if path_weights != "None":
    print("Loading weights from : " + path_weights)
    net.load_weights(path_weights)
else:
    sys.exit("<Error>: Specify a value for path_weights different from None when using test mode")

frames: Deque = deque(maxlen=nb_frames)
while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('<Error> can not capture video')

    frame = cv2.resize(frame, target_size)
    frames.append(frame)
    if len(frames) == nb_frames:
        res = net.predict(
            frames,
            verbose=1
        )
        print(res)
