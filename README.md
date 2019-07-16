[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

# WaifuLite
Super Resolution for Anime image. Lightweight implementation of Waifu2X (pytorch)

## Dependencies 
* [PyTorch](https://pytorch.org/) >= 1 ( > 0.41 shall also work, but not guarantee)
* [Nvidia/Apex](https://github.com/NVIDIA/apex/) (used for mixed precision training, you may use the [python codes](https://github.com/NVIDIA/apex/tree/master/apex/fp16_utils) directly)
* scipy, numpy, sklearn etc.

## Usage
- Training dataset: You may use any high resoluton/ high quality anime images as training data
- Training with MyTrain.py
