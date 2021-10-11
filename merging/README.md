# Merge the detectron2 pretrained model result and finetuned model result
## Implementation
- `merge_relations.ipynb`: to merge the relation generated from graph-rcnn
- `merge_data_simple_way.py`: to merge `attr_tables_with_target_box` in simple way (just drop rows from pretrained model result if there is an overlap with the result from finetuned model)
- `merge_data_considering_old_attr.py`: to merge `attr_tables_with_target_box` but use the attributes from pretrained model result if there is an overlap

## Notes
- if you use this merging, then you must use the file `rsa_merging_attr_old.py`  (rename it to rsa.py to replace the original one) to generate the expressions
