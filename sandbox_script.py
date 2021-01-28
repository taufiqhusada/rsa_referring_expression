import sys
import pandas as pd
import json
import numpy as np
from helper import *
import argparse
from rsa import RSA
import matplotlib.pyplot as plt
import os

with open('config.json') as config_file:
    config = json.load(config_file)
data_path = config['data_path']


file_id = 21540#3278#182

df = pd.read_csv(os.path.join(data_path,f'refCOCO/train/attr_tables/attr_{file_id}.tsv'), encoding='utf-8',sep='\t')

with open(os.path.join(data_path,f'refCOCO/train/labels/lab_{file_id}.json')) as json_file:
    label = json.load(json_file)
refs = [[r] for r in label['ref_sents']]
img_id = df['image_id'][0]
filename = os.path.join(data_path, f'refCOCO/train/imgs_by_id/{img_id}.jpg')
image = plt.imread(filename)
# get relations generated from graph faster-RCNN
rel_load = np.load(f'./train_relation_extraction.npy', allow_pickle=True)
generated_relations = rel_load[file_id]


box_data = df[['box_alias', 'x1','y1','w','h']]
fig,ax = plt.subplots(1)
img = image

# ax.imshow(img)
rng = [i for i in range(len(box_data))]
for i in [4]:#rng[:]:
    name, x,y,w,h = list(box_data.iloc[i,:])
    ax = draw_box_obj(name,x,y,w,h,img,ax)

print(label['ref_sents'])
bbox = label['bbox'][0]
sentence = label['ref_sents'][0]
fig,ax_true_label = plt.subplots(1)
ax_true_label.imshow(img)
draw_box_obj(sentence,bbox[0],bbox[1],bbox[2],bbox[3],img,ax_true_label)

rsa_agent = RSA(df, generated_relations=generated_relations)
rsa_agent.objects_by_type
# output = rsa_agent.full_speaker('woman-2')

matched_boxes = np.load('train_imgs_label_matching.npy', allow_pickle=True)[21540]

# print(output)
print("######")
for matched in matched_boxes:
    _, target,_ = matched
    print(target, rsa_agent.full_speaker(target))
print("$$$$$$")
print(rsa_agent.full_speaker('woman-2'))