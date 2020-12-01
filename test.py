import argparse
import configparser
import os
import sys
from ast import literal_eval

import cv2
import csv

from lib.data_loader import frame_queue
from lib.model_res import Resnet3DBuilder
from lib.data_loader import DataLoader

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
target_size: tuple = literal_eval(config.get('general', 'target_size'))
path_weights: str = config.get('path', 'path_weights')
data_vid: str = config.get('path', 'data_vid')
csv_labels: str = config.get('path', 'csv_labels')
path_vid: str = os.path.join(data_root, data_vid)
path_labels: str = os.path.join(data_root, csv_labels)

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,640)
cap.set(11,0)
cap.set(12,100)

# Loading the model
path_model = os.path.join(data_root, data_model, model_name)
inp_shape = (nb_frames,) + target_size + (3,)
net = Resnet3DBuilder.build_resnet_101(inp_shape, nb_classes)
if path_weights != "None":
    print("Loading weights from : " + path_weights)
    net.load_weights(path_weights)
else:
    sys.exit("<Error>: Specify a value for path_weights different from None when using test mode")
data = DataLoader(path_vid, path_labels)

with open('./data/csv_files/jester-v1-labels.csv') as f:
    f_csv = csv.reader(f)
    label_list = []
    for row in f_csv:
        label_list.append(row)
    label_list = tuple(label_list)

queue = frame_queue(nb_frames, target_size)
while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit('<Error> can not capture video')

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch_x = queue.img_inQueue(frame)
    res = net.predict(batch_x)
    res = list(res[0])
    index = res.index(max(res))
    print(label_list[index])
