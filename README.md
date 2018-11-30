# DISE-Domain-Invariant-Structure-Extraction
Pytorch Implementation of the paper All about Structure: Adapting Structural Information across Domains for Boosting Semantic Segmentation

## Paper
Ciation information here

## Introduction

## Installation

## Datasets

## Usage
### Train GTA5 to Cityscapes
```
python train_dise_gta2city.py --gta5_data_path path_to_gta5_dir --city_data_path path_to_cityscapes_dir
```
### More options
```
python train_dise_gta2city.py  -husage: train_dise_gta2city.py [-h] [--dump_logs DUMP_LOGS] [--log_dir LOG_DIR] [--gen_img_dir GEN_IMG_DIR]
                              [--gta5_data_path GTA5_DATA_PATH] [--city_data_path CITY_DATA_PATH]
                              [--data_list_path_gta5 DATA_LIST_PATH_GTA5]
                              [--data_list_path_city_img DATA_LIST_PATH_CITY_IMG]
                              [--data_list_path_city_lbl DATA_LIST_PATH_CITY_LBL]
                              [--data_list_path_val_img DATA_LIST_PATH_VAL_IMG]
                              [--data_list_path_val_lbl DATA_LIST_PATH_VAL_LBL]
                              [--cuda_device_id CUDA_DEVICE_ID [CUDA_DEVICE_ID ...]]

Domain Invariant Structure Extraction (DISE) for unsupervised domain adaptation for semantic segmentation
```

## Acknowledgement
