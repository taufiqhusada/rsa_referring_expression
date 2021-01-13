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
print(refs)
img_id = df['image_id'][0]
filename = os.path.join(data_path, f'refCOCO/train/imgs_by_id/{img_id}.jpg')
print(filename)
image = plt.imread(filename)

rsa_agent = RSA(df)

speech = rsa_agent.full_speaker('woman-1')

print(speech)