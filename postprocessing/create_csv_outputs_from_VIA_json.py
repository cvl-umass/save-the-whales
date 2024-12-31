import json
import pandas as pd
import numpy as np
import warnings
import copy
import argparse

warnings.filterwarnings("ignore")

all_csv_col_names = ["Image.ID",
"Whale.ID",
"Rostrum.X",
"Rostrum.Y",
"Peduncle.X",
"Peduncle.Y",
"Dorsal.Fin.Start.X",
"Dorsal.Fin.Start.Y",
"Dorsal.Fin.End.X",
"Dorsal.Fin.End.Y",
"Fluke.Endpoint.X1",
"Fluke.Endpoint.Y1",
"Fluke.Endpoint.X2",
"Fluke.Endpoint.Y2",
"Fluke.Middle.X",
"Fluke.Middle.Y",
"Blowhole.X",
"Blowole.Y",
"Eye.X1",
"Eye.Y1",
"Eye.X2",
"Eye.Y2",
"Total.Length",
"Area",
"Endpoint.Width.5.X1",
"Endpoint.Width.5.Y1",
"Endpoint.Width.5.X2",
"Endpoint.Width.5.Y2",
"Endpoint.Width.10.X1",
"Endpoint.Width.10.Y1",
"Endpoint.Width.10.X2",
"Endpoint.Width.10.Y2",
"Endpoint.Width.15.X1",
"Endpoint.Width.15.Y1",
"Endpoint.Width.15.X2",
"Endpoint.Width.15.Y2",
"Endpoint.Width.20.X1",
"Endpoint.Width.20.Y1",
"Endpoint.Width.20.X2",
"Endpoint.Width.20.Y2",
"Endpoint.Width.25.X1",
"Endpoint.Width.25.Y1",
"Endpoint.Width.25.X2",
"Endpoint.Width.25.Y2",
"Endpoint.Width.30.X1",
"Endpoint.Width.30.Y1",
"Endpoint.Width.30.X2",
"Endpoint.Width.30.Y2",
"Endpoint.Width.35.X1",
"Endpoint.Width.35.Y1",
"Endpoint.Width.35.X2",
"Endpoint.Width.35.Y2",
"Endpoint.Width.40.X1",
"Endpoint.Width.40.Y1",
"Endpoint.Width.40.X2",
"Endpoint.Width.40.Y2",
"Endpoint.Width.45.X1",
"Endpoint.Width.45.Y1",
"Endpoint.Width.45.X2",
"Endpoint.Width.45.Y2",
"Endpoint.Width.50.X1",
"Endpoint.Width.50.Y1",
"Endpoint.Width.50.X2",
"Endpoint.Width.50.Y2",
"Endpoint.Width.55.X1",
"Endpoint.Width.55.Y1",
"Endpoint.Width.55.X2",
"Endpoint.Width.55.Y2",
"Endpoint.Width.60.X1",
"Endpoint.Width.60.Y1",
"Endpoint.Width.60.X2",
"Endpoint.Width.60.Y2",
"Endpoint.Width.65.X1",
"Endpoint.Width.65.Y1",
"Endpoint.Width.65.X2",
"Endpoint.Width.65.Y2",
"Endpoint.Width.70.X1",
"Endpoint.Width.70.Y1",
"Endpoint.Width.70.X2",
"Endpoint.Width.70.Y2",
"Endpoint.Width.75.X1",
"Endpoint.Width.75.Y1",
"Endpoint.Width.75.X2",
"Endpoint.Width.75.Y2",
"Endpoint.Width.80.X1",
"Endpoint.Width.80.Y1",
"Endpoint.Width.80.X2",
"Endpoint.Width.80.Y2",
"Endpoint.Width.85.X1",
"Endpoint.Width.85.Y1",
"Endpoint.Width.85.X2",
"Endpoint.Width.85.Y2",
"Endpoint.Width.90.X1",
"Endpoint.Width.90.Y1",
"Endpoint.Width.90.X2",
"Endpoint.Width.90.Y2",
"Endpoint.Width.95.X1",
"Endpoint.Width.95.Y1",
"Endpoint.Width.95.X2",
"Endpoint.Width.95.Y2",
"width_5",
"width_10",
"width_15",
"width_20",
"width_25",
"width_30",
"width_35",
"width_40",
"width_45",
"width_50",
"width_55",
"width_60",
"width_65",
"width_70",
"width_75",
"width_80",
"width_85",
"width_90",
"width_95",]

def calculate_width(x1, y1, x2, y2):
    width = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return width

def get_width_at_percentage(csv_dict, percentage):
    feature_name = f'Endpoint.Width.{percentage}'
    if np.isnan(csv_dict[feature_name+'.X1']) or np.isnan(csv_dict[feature_name+'.Y1']) or np.isnan(csv_dict[feature_name+'.X2']) or np.isnan(csv_dict[feature_name+'.Y2']):
        width = np.nan
    else:
        width = calculate_width(csv_dict[feature_name+'.X1'], 
                           csv_dict[feature_name+'.Y1'], 
                           csv_dict[feature_name+'.X2'], 
                           csv_dict[feature_name+'.Y2'])
    return width

def main():
    csv_dict = {}

    for feature_name in all_csv_col_names:
        csv_dict[feature_name] = np.nan

    data = []
    print(f"Reading {args.input}")

    with open(args.input, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    data = data[0]['_via_img_metadata']

    print("Processing...")
    all_csv_data = []

    for img_name, img_all_points in data.items():
        csv_image_points = copy.deepcopy(csv_dict)
        image_name_without_file_type = ".".join(img_name.split(".")[:-1])
        csv_image_points['Image.ID'] = image_name_without_file_type+'.JPG'
        for img_point in img_all_points['regions']:
    #         print(img_point)
            if 'feature' not in img_point['region_attributes'].keys() or 'whale_id' not in img_point['region_attributes'].keys():
                continue
            feature_name = img_point['region_attributes']['feature']
            whale_id = img_point['region_attributes']['whale_id']
            x_coord = img_point['shape_attributes']['cx']
            y_coord = img_point['shape_attributes']['cy']
            
            csv_image_points['Whale.ID'] = whale_id
            
            if '_' in feature_name: ## Checking for endpoints
                feature_name_split = feature_name.split("_")
                side = feature_name_split[0][-1]
                if feature_name_split[-1].isnumeric():
                    percentage_width = feature_name_split[-1]
                    csv_feature_name = f'Endpoint.Width.{percentage_width}'
                else:
                    if feature_name_split[0][-1]=='fluke':
                        csv_feature_name = f'Fluke.Endpoint'
                    elif feature_name_split[0][-1]=='eye':
                        csv_feature_name = f'Eye'
                csv_image_points[f'{csv_feature_name}.X{side}'] = x_coord
                csv_image_points[f'{csv_feature_name}.Y{side}'] = y_coord
            elif feature_name =='rostrum':
                csv_image_points['Rostrum.X'] = x_coord
                csv_image_points['Rostrum.Y'] = y_coord
            elif feature_name =='fluke':
                csv_image_points['Fluke.Middle.X'] = x_coord
                csv_image_points['Fluke.Middle.Y'] = y_coord
            elif feature_name == 'blowhole':
                csv_image_points['Blowhole.X'] = x_coord
                csv_image_points['Blowhole.Y'] = y_coord
            elif feature_name == 'peduncle':
                csv_image_points['Peduncle.X'] = x_coord
                csv_image_points['Peduncle.Y'] = y_coord
            elif feature_name == 'dorsal_fin_start':
                csv_image_points['Dorsal.Fin.Start.X'] = x_coord
                csv_image_points['Dorsal.Fin.Start.Y'] = y_coord
            elif feature_name == 'dorsal_fin_end':
                csv_image_points['Dorsal.Fin.End.X'] = x_coord
                csv_image_points['Dorsal.Fin.End.Y'] = y_coord
        
        for i in range(5,100, 5):
            csv_image_points[f'width_{i}'] = get_width_at_percentage(csv_image_points, i)
 
        all_csv_data.append(csv_image_points)
 
    df = pd.DataFrame.from_dict(all_csv_data)

    df.to_csv(args.output)
    print("Done. Saved to", args.output)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process JSON from VIA and create CSV output file')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file from VGG Image Annotator')
    parser.add_argument('-o', '--output', required=True, help='Output CSV filename')
    args = parser.parse_args()
    
    main()