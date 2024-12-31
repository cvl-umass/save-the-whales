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
import glob
import time
import math
from collections import defaultdict
import datetime
import warnings
import copy
from numpy.random import default_rng
from PIL import Image
import copy
import pickle as pkl
import argparse

warnings.filterwarnings("ignore")

PREDICTED_CSV_DIR = './data_for_post_processing_by_students/predicted_csv_for_all_images'

def get_final_json_dict(df_):    
    final_dict = {}
    random_numbers = np.random.choice(range(100001, 999999), size=df_.shape[0], replace=False)
    
    shape_attributes_dict = {'name': 'point', 'cx':None, 'cy':None}
    # all_shape_attributes_dict = {}
    region_attributes_dict = {'feature': None, 'whale_id':None}
    regions_dict = {'shape_attributes':None,'region_attributes':None}
    all_file_names = []
    keypoint_names = ['Rostrum', 'Fluke']

    for i in range(df_.shape[0]):
    # for i in range(1):
        df_row = df_.iloc[i]
        rostrum_fluke_x  = (df_row['Rostrum.X'], df_row['Fluke.X'])
        rostrum_fluke_y = (df_row['Rostrum.Y'], df_row['Fluke.Y'])

        z = np.polyfit(rostrum_fluke_x,rostrum_fluke_y,1)

        temp_all_shape_attributes_dict = {}
        final_all_shape_attributes_dict = {}
        all_points_for_single_row = []
        final_dict_for_one_row = {}
        for col in df_row.index:
            split_col_name = col.split(".")
            if split_col_name[0]=='Endpoint':
                if not np.isnan(df_row[col]):
                    percentage_value = int(split_col_name[-2])
                    side_value = int(split_col_name[-1][-1])
                    if side_value == 1:
                        other_side_value = 2
                    else:
                        other_side_value = 1
                    if f'{str(percentage_value)}_{str(side_value)}' not in temp_all_shape_attributes_dict.keys():
                        shape_attributes_dict_new = copy.deepcopy(shape_attributes_dict)
                        temp_all_shape_attributes_dict[f'{str(percentage_value)}_{str(side_value)}'] = shape_attributes_dict_new

                    coord_type = split_col_name[-1][0]
                    if coord_type=='X':
                        shape_attributes_dict_new['cx'] = df_row[col].astype('float64')
                    else:
                        shape_attributes_dict_new['cy'] = df_row[col].astype('float64')
                    if shape_attributes_dict_new['cx'] and shape_attributes_dict_new['cy']:
                        value = np.polyval(z, shape_attributes_dict_new['cx']) - shape_attributes_dict_new['cy']
                        if (value>0 and side_value==1) or (value<0 and side_value==2):
                            final_all_shape_attributes_dict[f'{str(percentage_value)}_{str(side_value)}'] = shape_attributes_dict_new
                        else:
                            # print("Flipped")
                            final_all_shape_attributes_dict[f'{str(percentage_value)}_{str(other_side_value)}'] = shape_attributes_dict_new


        for key, val in final_all_shape_attributes_dict.items():
            split_key = key.split('_')
            feature_name = f'side{split_key[1]}_{split_key[0]}'
            region_attributes_dict_new = copy.deepcopy(region_attributes_dict)
            region_attributes_dict_new['feature'] = feature_name
            region_attributes_dict_new['whale_id'] = df_row['Whale.ID'].astype('float64')
            regions_dict_new = copy.deepcopy(regions_dict)
            regions_dict_new['region_attributes'] = region_attributes_dict_new
            regions_dict_new['shape_attributes'] = val
            all_points_for_single_row.append(regions_dict_new)

        for keypoint_name in keypoint_names:
            regions_dict_new =  add_point_to_final_dict(df_row, keypoint_name)
            all_points_for_single_row.append(regions_dict_new)

        key_to_final_dict = (str(df_row['Image.ID'])+'.JPG'+str(random_numbers[i]))
        all_file_names.append(key_to_final_dict)
        final_dict[key_to_final_dict] = {}
        final_dict[key_to_final_dict]['regions'] = all_points_for_single_row
        final_dict[key_to_final_dict]['size'] = 6392770
        final_dict[key_to_final_dict]['filename'] = df_row['Image.ID']+'Whale.'+str(df_row['Whale.ID'])+'.WidthMarked.jpg'
    #     final_dict[key_to_final_dict]['filename'] = df_row['Image.ID']+'.JPG'
        final_dict[key_to_final_dict]['file_attributes'] = {}
    return all_file_names, final_dict

def add_point_to_final_dict(df_row, point_name):
    shape_attributes_dict = {'name': 'point', 'cx':None, 'cy':None}
    region_attributes_dict = {'feature': None, 'whale_id':None}
    regions_dict = {'shape_attributes':None,'region_attributes':None}
    
    for col in df_row.index:
        split_col_name = col.split(".")
        if split_col_name[0]==point_name:
#             print(point_name)
            coord_type = split_col_name[-1][0]
            if coord_type=='X':
                    shape_attributes_dict['cx'] = df_row[col].astype('float64')
            else:
                shape_attributes_dict['cy'] = df_row[col].astype('float64')
            region_attributes_dict['feature'] = point_name.lower()
            region_attributes_dict['whale_id'] = df_row['Whale.ID'].astype('float64')
            if shape_attributes_dict['cx'] and shape_attributes_dict['cy']:
                break
    regions_dict['shape_attributes'] = shape_attributes_dict
    regions_dict['region_attributes'] = region_attributes_dict
    
    return regions_dict

def save_dict_to_correct_json_format(all_filenames, final_dict, dest_json_filename):
    dict_in_save_project_format = {"_via_settings":
                               {"ui":
                                {"annotation_editor_height":25,
                                 "annotation_editor_fontsize":0.70000000000000,
                                 "leftsidebar_width":20,
                                 "image_grid":
                                 {"img_height":
                                  80,
                                  "rshape_fill":"none",
                                  "rshape_fill_opacity":0.3,
                                  "rshape_stroke":"yellow",
                                  "rshape_stroke_width":2,
                                  "show_region_shape":True,
                                  "show_image_policy":"all"},
                                 "image":
                                 {"region_label":"__via_region_id__",
                                  "region_color":"whale_type",
                                  "region_label_font":"10px Sans",
                                  "on_image_annotation_editor_placement":"NEAR_REGION"}},
                                "core":{"buffer_size":18,
                                        "filepath":{},
                                        "default_filepath":""},
                                "project":{"name":"whale_annotation_attributes_chhandak"}},
                               "_via_img_metadata": final_dict,
                               "_via_attributes":{"region":{"feature":{"type":"dropdown","description":"Mark all important points on the whale","options":{"side1_5":"","side1_10":"","side1_15":"","side1_20":"","side1_25":"","side1_30":"","side1_35":"","side1_40":"","side1_45":"","side1_50":"","side1_55":"","side1_60":"","side1_65":"","side1_70":"","side1_75":"","side1_80":"","side1_85":"","side1_90":"","side1_95":"","side2_5":"","side2_10":"","side2_15":"","side2_20":"","side2_25":"","side2_30":"","side2_35":"","side2_40":"","side2_45":"","side2_50":"","side2_55":"","side2_60":"","side2_65":"","side2_70":"","side2_75":"","side2_80":"","side2_85":"","side2_90":"","side2_95":"","rostrum":"","blowhole":"","peduncle":"", "side1_eye":"", "side2_eye":"", "dorsal_fin_start":"", "dorsal_fin_end":"", "side1_fluke":"", "side2_fluke":"", "fluke":""},"default_options":{}},"whale_type":{"type":"radio","description":"","options":{"adult":"","calf":""},"default_options":{"adult":True}}},"file":{"num_whales":{"type":"radio","description":"","options":{"1":"","2":"","3+":""},"default_options":{"1":True}}}},"_via_data_format_version":"2.0.10","_via_image_id_list":all_filenames}
    with open(dest_json_filename, 'w') as f:
        json.dump(dict_in_save_project_format, f) 

def main():
    full_df =  pd.DataFrame()

    filename = args.input
    print(f"Reading {filename}")
    full_df = pd.read_csv(filename)
    print("Processing...")
    all_filenames, final_dict = get_final_json_dict(full_df)
    save_dict_to_correct_json_format(all_filenames, final_dict, args.output) 
    print(f"Done. Saved to {args.output}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV outputs and create JSON for VGG Image Annotator.')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file from ML model')
    parser.add_argument('-o', '--output', required=True, help='Output JSON filename')
    args = parser.parse_args()
    
    main()