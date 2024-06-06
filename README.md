# pytorch study

## BASE ENV
```shell
conda create -n pt python=3.10 -y

conda activate pt
```

## MAC
```shell
# 安装 pytorch v1.12版本已经正式支持了用于mac m1芯片gpu加速的mps后端
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y

pip install -r requirements.txt
```

## WIN
```shell
# 安装 pytorch v1.12版本已经正式支持了用于mac m1芯片gpu加速的mps后端
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Linux
```shell
# 安装 pytorch v1.12版本已经正式支持了用于mac m1芯片gpu加速的mps后端
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Proxy
```shell
-i https://pypi.tuna.tsinghua.edu.cn/simple
```