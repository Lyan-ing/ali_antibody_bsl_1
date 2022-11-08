#!/bin/bash
script_full_path=$(dirname "$0")
cd $script_full_path

python3 variant_extract.py
python3 data_preprocess.py
python3 Affinity_label_completion.py
python3 Neutralization_label_completion.py
python3 feature_extract.py
python3 Affinity_train.py
python3 Neutralization_train.py
python3 predict.py
