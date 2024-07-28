import detectron2
import json
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import os
import sys
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import glob
import time
import math
from collections import defaultdict
import datetime
import warnings
import glob
import torch
import pickle as pkl
import pandas as pd
import json
import argparse

np.random.seed(42)

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm

from sklearn.model_selection import train_test_split

DATA_PATH = '' ## Add path to image folder here
MODEL_SAVE_DIR = '' ## Add directory where you want the trained model to be stored

warnings.filterwarnings("ignore")

from detectron2.utils.logger import setup_logger
setup_logger()


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--lr", type=float, default=0.00025, help="initial learning rate")

opt = parser.parse_args()

lr = opt.lr
batch_size = opt.batch_size

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
#             if idx >= num_warmup * 2 or seconds_per_img > 5:
#                 total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
#                 eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
#                 log_every_n_seconds(
#                     logging.INFO,
#                     "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
#                         idx + 1, total, seconds_per_img, str(eta)
#                     ),
#                     n=5,
#                 )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
#         total_losses_reduced = sum(loss for loss in metrics_dict.values())
#         return total_losses_reduced
#         print("Check here:", metrics_dict.keys())
        loss_keypoint = metrics_dict['loss_keypoint']
        
        return loss_keypoint
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        

class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")

        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1,LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                )
            ))
            return hooks
def plot_keypoints(df, filename, whale_id, show_img=True):
    im = cv2.imread(DATA_PATH, "Australia/2016/images/",filename)
    datapoint = df[(df['filename']==filename) & (df['whale_id']==whale_id)].iloc[0]
    #print(datapoint)
    datapoint= datapoint.replace('',0)
    points = list(datapoint)[4:]
    points = points[:2] + points[-8:]
    if len(datapoint)!=0:
        for i in range(0,len(points),2):
            im = cv2.drawMarker(im, (int(points[i]),int(points[i+1])), markerType=cv2.MARKER_CROSS, thickness=5, color=(0, 255, 0))
        if show_img:
            plt.figure(figsize=(12,8))
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    else:
        print("Data not found")
    return im

def find_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def find_bounding_box(x1,y1,x2,y2,distance,width_at_fluke):
    bounding_box_values=[]
    factor=5
    if x1>x2:
        right=x1+distance/(factor*1.8)-50
        left=x2-distance/factor+20
    else:
        right=x2+distance/factor-20
        left=x1-distance/(factor*1.8)+50
    if abs(y1-y2)>50:
        if y1>y2:
            bottom=y1+distance/factor+30
            top=y2-distance/factor-30
        else:
            bottom=y2+distance/factor+30
            top=y1-distance/factor-30
    else:
        if y1>y2:
            bottom=y1+distance/(factor*1.5)+30
            top=y2-distance/(factor*1.5)-30
        else:
            bottom=y2+distance/(factor*1.5)+30
            top=y1-distance/(factor*1.5)-30
    
    if bottom-top<width_at_fluke*1.5:
        difference=(width_at_fluke*1.5)-(bottom-top)
        bottom=bottom+difference+20
        top=top-difference-20
    
    return {'r':right,'l':left,'t':top,'b':bottom}

df = pd.read_csv('./data/all_rows_filled_with_img_width_and_length_with_white_calves_jul_2023.csv')

distance_rostrum_to_fluke=[]
for i in range(df.shape[0]):
    distance_rostrum_to_fluke.append(find_distance(df['rostrum_x'].iloc[i],df['rostrum_y'].iloc[i],df['fluke_x'].iloc[i],df['fluke_y'].iloc[i]))

width_at_eye = []
for i in range(len(df)):
    width_at_eye.append(find_distance(df['side1_eye_x'].iloc[i], df['side1_eye_y'].iloc[i], df['side2_eye_x'].iloc[i], df['side2_eye_y'].iloc[i]))

bounding_boxes={}
for i in range(len(df)):
    bounding_box=(
            find_bounding_box(df['rostrum_x'].iloc[i],df['rostrum_y'].iloc[i],
            df['fluke_x'].iloc[i],df['fluke_y'].iloc[i],distance_rostrum_to_fluke[i],
            width_at_eye[i]))
    bounding_box_final = [int(bounding_box['l']),int(bounding_box['t']), #int(bounding_box['r']),int(bounding_box['b'])]
                          int(abs(bounding_box['l']-bounding_box['r'])),
                          int(abs(bounding_box['t']-bounding_box['b']))]
    
    if df['filename'].iloc[i] in bounding_boxes.keys():
        bounding_boxes[df['filename'].iloc[i]][df['whale_id'].iloc[i]]=(bounding_box_final)
    else:
        bounding_boxes[df['filename'].iloc[i]] = {df['whale_id'].iloc[i]: bounding_box_final}
        
segmentation_masks={}
keypoints = {}
count = 0
for i in range(len(df)):
    df_row=df.loc[i]
    segmentation_masks['filename'] = {}
    keypoints['filename'] = {}
    cols_list=df.columns
    poly = []
    j=4
    while (j<=35):
        if j==20:
            poly.append(int(df_row[cols_list[40]]))
            j+=1
            #continue
        elif j==21:
            poly.append(int(df_row[cols_list[41]]))
            j+=1
            #continue
        else:
            poly.append(int(df_row[cols_list[j]]))
            j+=1
    if df['filename'].iloc[i] in segmentation_masks.keys():
        segmentation_masks[df['filename'].iloc[i]][df['whale_id'].iloc[i]] = poly
    else:
        segmentation_masks[df['filename'].iloc[i]] = {df['whale_id'].iloc[i] : poly}
    
    keypoints_df_row = [df_row['rostrum_x'],df_row['rostrum_y'],2,df_row['blowhole_x'],df_row['blowhole_y'],2.0,df_row['len_mid_x'],df_row['len_mid_y'],2.0,df_row['peduncle_x'],df_row['peduncle_y'],2.0,df_row['fluke_x'],df_row['fluke_y'],2.0]
#     keypoints_df_row = [df_row['rostrum_x'],df_row['rostrum_y'],2,df_row['fluke_x'],df_row['fluke_y'],2]
    if df['filename'].iloc[i] in keypoints.keys():
        keypoints[df['filename'].iloc[i]][df['whale_id'].iloc[i]] = keypoints_df_row
    else:
        keypoints[df['filename'].iloc[i]] = {df['whale_id'].iloc[i] : keypoints_df_row}

data=[]
new_data={}
filename_set = set()
count=0
for i in range(len(df)):
#     print(i)
#     try:
        d1={}
        if df['filename'].iloc[i] in filename_set:
            continue
        d1["image_id"]=i
        d1["file_name"]=os.path.join(DATA_PATH, "Australia/2016/images/"+df['filename'].iloc[i])
        d1["annotations"] = [0]*max(bounding_boxes[df['filename'].iloc[i]].keys())
        for whale_id, bounding_box in bounding_boxes[df['filename'].iloc[i]].items():
            d2 = {}
            d2["bbox"]= bounding_box
            d2["bbox_mode"]=1
            d2["category_id"]=0
#             d2['segmentation'] = [segmentation_masks[df['filename'].iloc[i]][whale_id]]
            d2['keypoints'] = keypoints[df['filename'].iloc[i]][whale_id]
            d1["annotations"][int(whale_id)-1] = d2 

        filename_set.add(df['filename'].iloc[i])
        data.append(d1)

for i in range(len(data)):
    annotations = data[i]['annotations']
    if 0 in annotations:
        annotations.remove(0)
    for a in annotations:
        if not a:
            print(i)
    data[i]['annotations'] = annotations


index = [x for x in range(len(data))]
np.random.shuffle(index)
train_index = index[:int(0.75*len(data))]
val_index  = index[int(0.75*len(data)): int(0.85*len(data))]
test_index = index[int(0.85*len(data)):]

train = []
test = []
val = []
for i in range(len(data)):
    if i in train_index:
        train.append(data[i])
    elif i in val_index:
        val.append(data[i])
    elif i in test_index:
        test.append(data[i])
        
for i in range(len(val)):
    filename = filename=".".join((val[i]['file_name'].split("/")[-1]).split('.')[:-1])+".JPG"
    width = df['Image.Width'][df['filename']==filename].iloc[0]
    length = df['Image.Length'][df['filename']==filename].iloc[0]
    val[i]['width'] =  width
    val[i]['height'] = length

for i in range(len(test)):
    filename = filename=".".join((test[i]['file_name'].split("/")[-1]).split('.')[:-1])+".JPG"
    width = df['Image.Width'][df['filename']==filename].iloc[0]
    length = df['Image.Length'][df['filename']==filename].iloc[0]
    test[i]['width'] =  width
    test[i]['height'] = length

def get_whale_dicts(d):
    if d=="train":
        return train
    elif d=="val":
        return val
    elif d=='test':
        return test

keypoint_names = ['rostrum', 'blowhole', 'len_mid', 'peduncle', 'fluke']
"""keypoint_names = ['rostrum', 'side1_head1', 'side1_head2',
                  'side1_eye', 'side1_body1', 'side1_body2', 'side1_body3',
                  'side1_peduncle_width','peduncle', 'side2_peduncle_width',
                  'side2_body3', 'side2_body2', 'side2_body1',
                  'side2_eye', 'side2_head2', 'side2_head1','blowhole', 'len_mid','fluke']"""
# keypoint_names = ['rostrum', 'fluke']
for d in ["train","test","val"]:
    DatasetCatalog.register("whales_" + d,lambda d=d: get_whale_dicts(d))
    MetadataCatalog.get("whales_" + d).set(thing_classes=["whale"])
    MetadataCatalog.get("whales_" + d).set(keypoint_names=keypoint_names)
    MetadataCatalog.get("whales_" + d).set(keypoint_flip_map=[])
#     MetadataCatalog.get("whales_" + d).set(keypoint_flip_map=[["fluke", "rostrum"]])
whales_metadata = MetadataCatalog.get("whales_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("whales_train",)
cfg.DATASETS.TEST = ("whales_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = lr 
cfg.SOLVER.MAX_ITER = 10 #2000   
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
cfg.TEST.EVAL_PERIOD = 500 
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
cfg.OUTPUT_DIR=os.path.join(MODEL_SAVE_DIR, f'keypoint_detection_final_lr_{lr}_batch_size_{batch_size}_epochs_2000_sbatch')
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((5, 1), dtype=float).tolist()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
