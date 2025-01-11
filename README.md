# AI-Learning

## BASE ENV

## MAC
```shell
conda create -n ail python=3.10 -y
conda activate ail
# 安装 pytorch v1.12版本已经正式支持了用于mac m1芯片gpu加速的mps后端
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y

# tensorflow mac apple silicon
pip install tensorflow-macos
pip install tensorflow-metal

pip install -r requirements.txt
```

## linux
```shell
conda create -n ail-tf python=3.9 -y
conda create -n ail-pt python=3.10 -y

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# tensorflow 需要cudnn支持
pip install tensorflow

pip install -r requirements.txt
```

