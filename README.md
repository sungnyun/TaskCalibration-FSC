# [Official]  Calibration of Few-Shot Classification Tasks

This repository contains the official PyTorch implementation of the following paper:
> **Calibration of Few-Shot Classification Tasks: Mitigating Misconfidence from Distribution Mismatch** by
> Sungnyun Kim and Se-Young Yun, IEEE _Access_ vol. 10, 2022, doi:10.1109/ACCESS.2022.31760902022.
> 
> **Paper**: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9777704
>
> **Abstract:** *As many meta-learning algorithms improve performance in solving few-shot classification problems for practical applications, the accurate prediction of uncertainty is considered essential. In meta-training, the algorithm treats all generated tasks equally and updates the model to perform well on training tasks. During the training, some of the tasks may make it difficult for the model to infer the query examples from the support examples, especially when a large mismatch between the support set and the query set exists. The distribution mismatch causes the model to have incorrect confidence, which causes a calibration problem. In this study, we propose a novel meta-training method that measures the distribution mismatch and enables the model to predict with more precise confidence. Moreover, our method is algorithm-agnostic and can be readily expanded to include a range of meta-learning models. Through extensive experimentation, including dataset shift, we show that our training strategy prevents the model from becoming indiscriminately confident, and thereby helps the model to produce calibrated classification results without the loss of accuracy.*

<p align="center">
  <img src="/img/overview.png" width="500">    
</p>

## Table of Contents

* [Prerequisites](#prerequisites)
* [Data Setup](#data-setup)
* [Usage](#usage)
* [Citing This Work](#citing-this-work)
* [License](#license)

## Prerequisites

Our code works on `torch>=1.5`. Install the required Python packages via

```sh
pip install -r requirements.txt
```

## Data Setup
Make sure that the dataset directories are correctly specified in `path_configs.py`, and that your dataset directories contain the correct files.

### miniImageNet
In `./filelists/miniImagenet/`, there is a shell script to setup the dataset files.
```sh
bash download_miniImagenet.sh
```
### CUB
In `./filelists/CUB/`, there is a shell script to setup the dataset files.
```sh
bash download_CUB.sh
```
### tieredImageNet
In `./filelists/tieredImagenet/`, there is a python file. In the python file, there is an instruction how to download and place the dataset.
After setup, run the python file.
```sh
python write_tieredImagenet_filelist.py
```
## Usage

The main training file is `train.py` and `test.py`. To see all CLI arguments, refer to `utils.py`.    
The following command lines will reproduce the results.

1. miniImageNet 5-way 5-shot TCMAML:
```sh
python train.py --dataset miniImagenet
```
2. miniImageNet 5-way 5-shot TCMAML (LS)
```sh
python train.py --dataset miniImagenet --linear-scaling
```
3. miniImageNet 5-way 5-shot TCProtoNet
```sh
python train.py --dataset miniImagenet --method tcproto
```
4. miniImageNet 5-way 1-shot TCMAML
```sh
python train.py --dataset miniImagenet --n-shot 1
```
5. CUB 5-way 5-shot TCMAML:
```sh
python train.py --dataset CUB
```
6. miniImageNet -> CUB (dataset shift) 5-way 5-shot TCMAML
```sh
python train.py --dataset cross --model ResNet18
```
7. miniImageNet 10-way 5-shot with Conv6 backbone TCMAML
```sh
python train.py --dataset miniImagenet --model Conv6 --train-n-way 10 --test-n-way 10
```

Other important arguments include `--temp`, `--stop-epoch`, and `--corrupted-task`.

To evaluate and check the calibration results after training is done, run `test.py` with the same arguments.   
For exmaple, after training miniImageNet 5-way 1-shot TCProtoNet, run
```sh
python test.py --dataset miniImagenet --n-shot 1 --method tcproto
```

## Citing This Work

If you find this repo useful for your research, please consider citing our paper:
```
@article{kim2022calibration,
  title={Calibration of Few-Shot Classification Tasks: Mitigating Misconfidence from Distribution Mismatch},
  author={Kim, Sungnyun and Yun, Se-Young},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
```

## License

Distributed under the MIT License.
