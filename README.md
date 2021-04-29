# rsa_referring_expression

The project contains an implementation of the RSA framework applied to the refCOCO dataset and the scenegraph.

## Data preparation
  -download the refCOCO dataset (test dataset) to your desired location
  - the `refCOCO` folder should contain the following directories:
    - attr_tables
    - attr_tables_with_target_box
    - imgs_by_id
    - labels
    - rel_tables
  - update the path in the `config.json` file to your location, the path should be an absolute path to the `refCOCO` folder.
  
## Execute the code
  - To run a demo, try the `Demo_compute_expression_with_rsa.ipynb` notebook.
  - To generate the expressions for some images, run `python calc_expression.py --start <start_index> --end <end_index> --k <top k guesses> --target_set test` where `start` and `end` are the start and end index of image reference id we want to process and `k` is the top-k guess that we want to find expression (the default value for k is 3)
  - To calculate the Bleu & Rouge score, check the `Compute_Bleu_Rouge_score.ipynb`
  - To try the the object matching algorithm, run the `Process_refCOCO_images_and_match_object_to_label.ipynb`


## Implementation
  - The `rsa` implementation is in the `rsa.py` file
  - The `helper.py` contains helper functions used to extract matching objects from the scenegraph output.
