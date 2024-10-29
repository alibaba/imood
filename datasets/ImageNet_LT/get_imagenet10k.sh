#!/bin/bash

DATA=./data

wget https://image-net.org/data/imagenet10k_eccv2010.tar 
mkdir ${DATA}/imagenet10k
tar -xvf imagenet10k_eccv2010.tar -C ${DATA}/imagenet10k
python datasets/ImageNet_LT/select_extra_imagenet_1k.py --drp ${DATA}
python datasets/ImageNet_LT/extract_extra_imagenet_1k.py --drp ${DATA}
