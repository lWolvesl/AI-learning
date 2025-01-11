#!/bin/bash
# nvcr.io/nvidia/tensorflow:24.12-tf2-py3
docker run -itd --gpus all --name ail-tf -v /data/szh2/Project/AI-learning:/workspace/ail -p 58888:8888 nvcr.io/nvidia/tensorflow:24.12-tf2-py3 jupyter notebook --NotebookApp.token='12345678'
