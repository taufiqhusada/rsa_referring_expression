import json
import pandas as pd
import os
from collections import defaultdict
import sys
sys.path.append('../')
from helper import *

merged_data_path = '/scratch2/thdaryan/data/merged_result_using_attr_old_data'
with open('../config.json') as config_file:
    config = json.load(config_file)
old_data_path = config['old_data_path']

print('old_data_path: ', old_data_path)
# result data from fine tuned detectron
new_data_path = '/scratch2/thdaryan/data/fine_tuned'

def merge_old_and_new_df(file_idx):
    # dict list subclass (still hardcoded)
    dict_list_subclass = {'person':['man', 'guy', 'boy', 'girl', 'woman'], 'man': ['boy', 'guy'], 'woman':['woman', 'girl', 'bride']}
    
    df_old = pd.read_csv(os.path.join(old_data_path,f'refCOCO/test/attr_tables_with_target_box/attr_{file_idx}.tsv'), encoding='utf-8',sep='\t')
    df_new = pd.read_csv(os.path.join(new_data_path,f'refCOCO/test/attr_tables_with_target_box/attr_{file_idx}.tsv'), encoding='utf-8',sep='\t')
    
    # Preprocess
    # drop type_oov in df_new because it is already on df_old
    df_new.drop(['TYPE_OOV'], axis=1, inplace = True)
    
    # drop type none in new data
    list_col = df_new.columns
    for col in list_col:
        if(len(col) >= len('TYPE_None') and col[:len('TYPE_None')]=='TYPE_None'):
            df_new.drop([col], axis=1, inplace = True)
    
    list_type_old_data = [col for col in df_old.columns if 'TYPE_' in col]
    list_type_new_data = [col for col in df_new.columns if 'TYPE_' in col]
    
    # add flag to identify it is the old data or new data
    df_old['is_old_type'] = True
    df_old['is_old_attr'] = True

    df_new['is_old_type'] = False
    df_new['is_old_attr'] = False

    
    # Make the columns in old df and new df same
    for col in list_type_old_data:
        if (col not in list_type_new_data):
            df_new[col] = 0

    for col in list_type_new_data:
        if (col not in list_type_old_data):
            df_old[col] = 0
            
    # Merge old df and new df
    # get all class name in old data and save the index
    df_merged = df_old.copy()

    dict_class_list_index = defaultdict(list)
    for i, row in df_old[['box_alias']].iterrows():
        class_name = row['box_alias'][:row['box_alias'].find('-')]
        dict_class_list_index[class_name].append(i)

    list_idx_to_be_dropped = []
    for i, row in df_new.iterrows():
        class_name = row['box_alias'][:row['box_alias'].find('-')]

        if (class_name in dict_class_list_index):
            list_idx_to_be_dropped += dict_class_list_index[class_name]

        # handle sub-class (still hardcoded)
        if (class_name in dict_list_subclass):
            for subclass_name in dict_list_subclass[class_name]:
                list_idx_to_be_dropped += dict_class_list_index[subclass_name]


        # handle overlap boxes (choose attribute from old data instead)
        # find overlap
        result_top_match = None
        result_top_5_match = top_5_match(df_old[['box_alias', 'x1','y1','w','h']], row[5:9])
#         print(result_top_5_match)
        for res in result_top_5_match:
            idx_old = res[0]
            name = res[1]
            similarity = res[2]
            if (name.split('-')[0]==class_name or (class_name in dict_list_subclass and name.split('-')[0]==dict_list_subclass[class_name])):
                result_top_match = (idx_old, name, similarity)
                break
                
        if (result_top_match != None):
            if (similarity > .6):
                overlap_obj_idx = result_top_match[0]

                # use attribute from old data
                for col in df_old.columns:
                    if ('ATTR' in col):
                        row[col] = df_old.loc[overlap_obj_idx][col]
                row['is_old_attr'] = True
        
        df_merged = df_merged.append(row, ignore_index = True)
       

    set_idx_to_be_dropped = set(list_idx_to_be_dropped)   #just removing some duplicates

    df_merged = df_merged.drop(set_idx_to_be_dropped).reset_index(drop=True)
    
    # check if there is NaN value (if so then there is some error on the process)
    is_nan = False
    for col in df_merged.columns:
        is_nan = df_merged[col].isnull().values.any()
    if (is_nan):
        raise Exception('there is NaN value')
        
    # save the result
    df_merged.to_csv(os.path.join(merged_data_path, f'refCOCO/test/attr_tables_with_target_box/attr_{file_idx}.tsv'), sep = '\t', index=False)
    
for i in range(5000):
    merge_old_and_new_df(i)
    if (i % 50 == 0 ):
        print(f'finished processing {i} files')
    
   