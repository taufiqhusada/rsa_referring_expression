import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from helper import *
import os
from rsa import RSA
import argparse
import random

with open('config.json') as config_file:
    config = json.load(config_file)
data_path = config['data_path']

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def generate_bounded_boxes(start=0, end=1, target_set='test'):
    label_box_ids = {}
    for file_id in range(start, end):
        df = pd.read_csv(os.path.join(data_path,f'refCOCO/{target_set}/attr_tables/attr_{file_id}.tsv'), encoding='utf-8',sep='\t')
        img_id = df['image_id'][0]
        filename = os.path.join(data_path, f'refCOCO/{target_set}/imgs_by_id/{img_id}.jpg')
        image = plt.imread(filename)
        with open(os.path.join(data_path,f'refCOCO/{target_set}/labels/lab_{file_id}.json')) as json_file:
            label = json.load(json_file)
        box_data = df[['box_alias', 'x1','y1','w','h']]
        bbox = label['bbox'][0]
        fig,ax = plt.subplots(1, figsize=(7,7))
        img = image
        colors = get_cmap(10)
        ax.imshow(img)
        rng = [i for i in range(len(box_data))]
        counter = 0
        boxes_to_add = []
        x1, y1, w1,h1 = bbox[0],bbox[1],bbox[2],bbox[3]
        for i in rng[:]:
            if counter >=5:
                break
            name, x,y,w,h = list(box_data.iloc[i,:])
            ####
            overlapped_area = calc_overlap(x1,y1,w1,h1,x,y,w,h)
            total_area = w1*h1 + w*h - overlapped_area
            similarity = overlapped_area/total_area
            if similarity > 0.5:
                continue
            ####
            boxes_to_add.append([name, x,y,w,h])
            counter += 1

        # DRAWING THE BOX OF THE TRUE LABEL AND PRINT THE TRUE LABELS (REFCOCO)
        boxes_to_add.append(['true obj',bbox[0],bbox[1],bbox[2],bbox[3]])
        random.shuffle(boxes_to_add)
        for i, v in enumerate(boxes_to_add):
            name, x,y,w,h = v
            if name == 'true obj':
                label_box_ids[file_id] = i
            ax, color = draw_box_obj(str(i),x,y,w,h,img,ax,color=colors(i))
        
        fig.savefig(f'./data/validation_images/{target_set}/validation_img_{file_id}.png')
        with open(f'./data/validation_images/{target_set}/label_test_data.json', 'w') as fp:
            json.dump(label_box_ids, fp)
        np.save(f'./data/validation_images/{target_set}/label_box_ids.npy', label_box_ids)
        if file_id % 50 == 0:
            print(f'finished file {file_id}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc_expression')
    parser.add_argument('--start', type=int, default=0, help='start file')
    parser.add_argument('--end', type=int, default=5, help='end file (not inclusive)')
    parser.add_argument('--target_set', type=str, default='test', help='either generate from the train or the test dataset, default is test')
    args = parser.parse_args()
    generate_bounded_boxes(start=args.start, end=args.end, target_set=args.target_set)