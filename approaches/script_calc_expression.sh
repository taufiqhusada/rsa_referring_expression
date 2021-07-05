#!/bin/sh

python calc_expression.py --start 0 --end 1000 --target_set test --top_k_features 3 --k 5
python calc_expression.py --start 1000 --end 2000 --target_set test --top_k_features 3 --k 5
python calc_expression.py --start 2000 --end 3000 --target_set test --top_k_features 3 --k 5
python calc_expression.py --start 3000 --end 4000 --target_set test --top_k_features 3 --k 5
python calc_expression.py --start 4000 --end 5000 --target_set test --top_k_features 3 --k 5