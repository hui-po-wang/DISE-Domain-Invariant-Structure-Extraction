# DISE-Domain-Invariant-Structure-Extraction
Pytorch Implementation of the paper All about Structure: Adapting Structural Information across Domains for Boosting Semantic Segmentation.

Introduction video: [youtube](https://youtu.be/YnD_zQNbfK4)
## Paper
Ciation information here

## Example Results
![prediction_results.png](examples/prediction_results.png)

## Quantitative Results
![performance.png](examples/performance.png)

## Prerequisite
- Pytorch 0.3.1  
- Nvidia GPU with at least 16 GB memory 

## Installation
```
git clone https://github.com/a514514772/DISE-Domain-Invariant-Structure-Extraction.git
```
## Datasets

1. Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) as the source domain and unzip it to  `/data`
2. Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com) as the target domain and unzip it to  `/data`

The structure of `/data` may look like this:
  ```
  ├── data
      ├── Cityscapes
      │   ├── gtFine
      │   └── leftImg8bit
      ├── GTA5
          ├── images
          └── labels
  ```
## Usage
### Pretrained Weights
[Google drive](https://drive.google.com/drive/folders/1NSPhGnTqBp6oeBd6awNs7VVTGuL1p4wP?usp=sharing)

### Example Training Script: GTA5 to Cityscapes
```
python train_dise_gta2city.py --gta5_data_path /data/GTA5 --city_data_path /data/Cityscapes
```
### Example Testing Script:
Note that, to test performance on the testing set, we provide scripts to generate 1024x2048 outputs which are compatible with [the testing server](https://www.cityscapes-dataset.com/benchmarks/#instance-level-results).
```
python evaluate.py ./weights --city_data_path /data/Cityscapes
```
### More options
```
python train_dise_gta2city.py  -h
usage: train_dise_gta2city.py [-h] [--dump_logs DUMP_LOGS] [--log_dir LOG_DIR] [--gen_img_dir GEN_IMG_DIR]
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
We implement this project heavily based on [AdaptSeg](https://github.com/wasidennis/AdaptSegNet) proposed by Tsai et el..
