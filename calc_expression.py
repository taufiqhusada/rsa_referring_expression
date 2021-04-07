import sys
import pandas as pd
import json
import numpy as np
from helper import *
import argparse
from rsa import RSA
import os
import tensorflow as tf

with open('config.json') as config_file:
    config = json.load(config_file)
data_path = config['data_path']


# start: the start index of the input file
# end: the end index of the input file(exclusive)
# k: top k guessed expressions
def calc_expression(start=0, end=5, k=3, target_set='test'):
    matched_label = np.load(f'{target_set}_imgs_label_matching_with_true_target.npy', allow_pickle=True)
    rel_load = np.load(f'./{target_set}_relation_extraction.npy', allow_pickle=True)
    exps = []
    references = []
    low_confidence_track = [] # keep track if rsa ever encounter a low confident situation (don't have anything to say or no preference-0probabilty everywhere)
    for i in range(start, end):
        df = pd.read_csv(os.path.join(data_path,f'refCOCO/{target_set}/attr_tables_with_target_box/attr_{i}.tsv'), encoding='utf-8',sep='\t')

        # UNCOMMENT TO SAVE THE REFERENCES OF THE SAME RANGE AS THE PROCESSED IMAGES
        with open(os.path.join(data_path,f'refCOCO/{target_set}/labels/lab_{i}.json')) as json_file:
            label = json.load(json_file)
        refs = [[r] for r in label['ref_sents']]
        references.append(refs)
        generated_relations = rel_load[i]
        
        ### adding LSTM to adjust utterance prior ###
        #lstm = tf.keras.models.load_model('three_gram_lstm_loss_2.8469_accuracy_0.4227.h5')
#         f = open('word_to_idx_vice_versa.json')
#         tokenizer = json.load(f)
#         word_to_idx = tokenizer['word_to_idx']
#         idx_to_word = tokenizer['idx_to_word']
        
        rsa_agent = RSA(df, generated_relations=generated_relations)#,\
                        #model=lstm, word_to_idx=word_to_idx, idx_to_word=idx_to_word)
        targets = [matched_label[i][j][1] for j in range(min(k, len(matched_label[i])))]
        word_lists = []
        is_low_confidence = []
        for target in targets:
            list_utterances, low_confidence = rsa_agent.full_speaker(target)
            word_lists.append(list_utterances)
            is_low_confidence.append(low_confidence)
        expression = [' '.join(word_lists[j][::-1]) for j in range(len(word_lists))]
        exps.append(expression)
        low_confidence_track.append(is_low_confidence)
        if i % 50 == 0:
            print(f'finished file {i}')

    np.save(f'./data/{target_set}/detectron2_with_target/top{k}_exps_from_{start}_to_{end}.npy',exps)
    np.save(f'./data/{target_set}/detectron2_with_target/top{k}_exps_confidence_record_from_{start}_to_{end}.npy',low_confidence_track)
    # UNCOMMENT TO SAVE THE REFERENCES OF THE SAME RANGE AS THE PROCESSED IMAGES
    np.save(f'./data/{target_set}/detectron2_with_target/references_from_{start}_to_{end}.npy',references)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc_expression')
    parser.add_argument('--start', type=int, default=0, help='start file')
    parser.add_argument('--end', type=int, default=5, help='end file (not inclusive)')
    parser.add_argument('--k', type=int, default=3, help='top k guesses, maximum is 5')
    parser.add_argument('--target_set', type=str, default='test', help='either generate from the train or the test dataset, default is test')
    args = parser.parse_args()
    start, end = args.start, args.end
    k = args.k
    target_set = args.target_set
    calc_expression(start, end, k, target_set)