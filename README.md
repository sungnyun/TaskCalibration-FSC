# Calibration of Few-Shot Classification Tasks
This repo contains the implementation of the paper,    
**Calibration of Few-Shot Classification Tasks: Mitigating Misconfidence from Distribution Mismatch**    
that has been published in _IEEE Access_, vol. 10, 2022, doi:10.1109/ACCESS.2022.3176090.

<center><img src="/img/overview.png" width="70%" height="70%"></center>

## Table of Contents

* [Prerequisites](#prerequisites)
* [Data Setup](#data-setup)
* [Usage](#usage)
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

## License

Distributed under the MIT License.
