# DL-Final-Project


## Description
This repository is final-project of CCU_DL. Given single image or video, the goal is detect and recognize category of merchandise on shelf. 
<br/>
<br/>

## Install
### Clone the repository recursively
    git clone --recurse-submodules https://github.com/Yenhongxuan/DL-Final-Project.git
If you have already cloned and forgot to use `--recurse-submodules`, run `git submodule updata --init`

### Build environment
Step 1.

    conda create -n env_name python=3
Step 2.

    pip3 install -r requirements.txt

<br/>
<br/>


## Datasest
Link below is dataset we used. We collect these image by downloading from internet and taking photo at convenience store. 

https://drive.google.com/drive/folders/1HWXsozNFpdiMKPjGbrHirxvPiqkWc50o?usp=sharing

Our dataset is split to six categories, which are beverage, bread, cookie, frozen-food, riceball, instant-noodles



## Training
### YOLOv5
You can excute like the following to train YOLOv5

    cd yolov5/
    python train.py --data SKU-110K.yaml --weights yolov5n.pt --img 640
The weight will be stored in `./yolov5/runs/train/exp/weights/`
<br/>
<br/>
### DenseNet
Options for train.py are as following

`--root` Root directory of data

`--epochs` Number of epoch to train

`--lr` Learning rate

`--bs` Batchsize

`--balance_sample` Whether apply weighted sampleing during training
    python3 train.py

You can excute the following to train Grocery recognition network

    python3 train.py --root 'data_fir' --balance_sample
## Inference
Options for inference.py are as following:

`--source` Path of image or video to be inferenced

`--weights` Weight of YOLO to be used

`--best_model` Path of Check-point of ResNet model

`--num_classes` Amount of category to be predicted

You can excute like this

    python3 inference.py --source ./sample_img/1.jpg --weights ./yolo_weights/yolo_best.pt --best_model ./ResNet_weight/best_model_1.pt
