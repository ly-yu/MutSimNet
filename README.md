# MutSimNet
Part of the code involved in this paper has been uploaded.

# 目录

[View English](./README.md)

<!-- TOC -->

- [Contents](#Contents)
- [MutSimNet](#MutSimNet)
- [Model](#Model)
- [Datasets](#Datasets)
- [Evaluation Metrics](#Metrics)
    - [F1](#F1)
    - [recall](#recall)
    - [IoU](#IoU)
    - [precision](#precision)
- [Environment](#Environment)
- [Quick start](#start)
- [Code](#Code)
    - [Scripts](#Scripts)
    - [Parameters](#Parameters)
    - [Train](#Train)
        - [Train](#train)
    - [Predict](#Predict)
        - [Predict](#Predict)
- [Model](#Model)
    - [Performance](#Performance)
        - [Train](#Train)
            - [Levir_CD](#Training)
            - [S2Looking](#Training)
        - [Val](#Val)
            - [Levir_CD](#Verifying)
            - [S2Looking](#Verifying)
    
<!-- /TOC -->

# MutSimNet

Part of the code involved in the paper has been uploaded.

# Paper

[Paper](None) : Mutually Reinforcing Similarity Learning for RS Image Change Detection

# Model

[Model Weights]() : Weight files saved by model training.

# Datasets

Dataset : [Levir_CD](https://justchenhao.github.io/LEVIR/)

- Data format：.png
    - Note: the data is saved in PaddleCD/data.

Dataset : [S2Looking](https://github.com/S2Looking/Dataset)

- Data format：.png
    - Note: the data is saved in PaddleCD/data.
  

# Evaluation Metrics

## F1, Iou, Recall, precision

F1, Iou, Recall and precision are adopted as evaluation metrics of model training.

# Environment

- Hardware（GPU）
    - Using GPU processor to build hardware environment.
- Framework
    - [PaddlePaddle](https://www.paddlepaddle.org.cn/)

# Quick start

After installing PaddlePaddle through the official website, you can follow the following steps for training and evaluation:

- GPU processor runs in 32G environment.

  ```yaml
  batch_size: 8
  iters: 80000

  train_dataset:
    # the type of dataset
    # path: PaddleCD\paddleseg\datasets\levir_cd.py
    type: LevirCD
    # Training set path
    dataset_root: data/train_split
  ....
  val_dataset:
    type: LevirCD
    # validation set path
    dataset_root: data/test
  ....
  ```
  
 ```python
  # Frame preparation
  !cd PaddleCD
  
  !pip install -r requirements.txt --user
  
  !pip install -v -e .            # This line needs to be re-run if the code is modified, but not if the configuration file is changed.
  
  !python setup.py install
 ```

```python
  # train
  python tools/train.py --config configs/CD/Swin_Tiny_CD_Sub.yml --save_interval 800 --use_vdl --save_dir output/UperNet_Sub --log_iters 50 --num_worker 2 --do_eval --precision fp16 --amp_level O1

  # val
  python tools/val.py --config configs/CD/Swin_Tiny_CD_Sub.yml --model_path output/UperNet_Sub/best_model/model.pdparams --aug_eval --scales 0.75 1.0 1.25 --flip_horizontal --is_slide --crop_size 512 512 --stride 256 256

  # predict
  python predict.py --config configs/CD/Swin_Tiny_CD_Sub.yml --model_path output/UperNet_Sub/best_model/model.pdparams --image_path data/test/ --save_dir output/result --custom_color 0 0 0 255 255 255
```

# Code

## Code

    ```bash
    ├── MutSimNet
        ├── Preparation.ipynb         
        ├── PaddleCD
        │   ├──configs                   
        │   │   ├──CD
        │   ├──paddleseg                   
        │   │   ├──CD           
        │   ├──contrib               
        │   ├──data          
        │   ├──deploy          
        │   ├──docs               
        │   ├──EISeg          
        │   ├──Matting
        │   ├──paddleseg.egg-info
        │   ├──test_tipc
        │   ├──tests
        │   ├──tools
        │   ├──requirements.txt
        │   ├──setup.py
        ├── README.md              
    '''


## Train

### Train

- GPU processor runs in 32G environment.

    '''python

        python tools/train.py --config --save_interval --use_vdl --save_dir --log_iters --num_worker --do_eval --precision --amp_level
    '''

##Val
###Val
- Evaluation at runtime in GPU environment.

    '''python

        python tools/val.py --config --model_path 
    '''
