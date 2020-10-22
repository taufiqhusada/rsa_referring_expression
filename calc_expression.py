import sys
import pandas as pd
import json
import numpy as np
from helper import *
import argparse
from rsa import RSA
import os

with open('config.json') as config_file:
    config = json.load(config_file)
data_path = config['data_path']


# start: the start index of the input file
# end: the end index of the input file(exclusive)
# k: top k guessed expressions
def calc_expression(start=0, end=5, k=3):
    matched_label = np.load('test_imgs_label_matching.npy', allow_pickle=True)

    exps = []
    references = []
    for i in range(start, end):
        df = pd.read_csv(os.path.join(data_path,f'refCOCO/attr_tables/attr_{i}.tsv'), encoding='utf-8',sep='\t')

        # UNCOMMENT TO SAVE THE REFERENCES OF THE SAME RANGE AS THE PROCESSED IMAGES
        with open(os.path.join(data_path,f'refCOCO/labels/lab_{i}.json')) as json_file:
            label = json.load(json_file)
        refs = [[r] for r in label['ref_sents']]
        references.append(refs)
        rsa_agent = RSA(df)
        targets = [matched_label[i][j][1] for j in range(min(k, len(matched_label[i])))]
        word_lists = [rsa_agent.full_speaker(target) for target in targets]
        expression = [' '.join(word_lists[j][::-1]) for j in range(len(word_lists))]
        exps.append(expression)
        if i % 50 == 0:
            print(f'finished file {i}')

    np.save(f'top{k}_exps_from_{start}_to_{end}.npy',exps)
    # UNCOMMENT TO SAVE THE REFERENCES OF THE SAME RANGE AS THE PROCESSED IMAGES
    np.save(f'references_from_{start}_to_{end}.npy',references)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc_expression')
    parser.add_argument('--start', type=int, default=0, help='start file')
    parser.add_argument('--end', type=int, default=5, help='end file (not inclusive)')
    parser.add_argument('--k', type=int, default=3, help='top k guesses, maximum is 5')
    args = parser.parse_args()
    start, end = args.start, args.end
    k = args.k
    calc_expression(start, end, k)