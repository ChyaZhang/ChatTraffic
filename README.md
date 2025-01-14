# ChatTraffic
Official PyTorch implementation of [ChatTraffic: Text-to-Traffic Generation via Diffusion Model](https://ieeexplore.ieee.org/document/10803279)

[![arXiv](https://img.shields.io/badge/arXiv-2311.16203-b31b1b.svg)](https://arxiv.org/abs/2311.16203)

## Introduction

ChatTraffic is capable of generating traffic situations (speed, congestion level, and travel time) according to the text. This enables ChatTraffic to provide predictions of how future events (road construction, unexpected accidents, unusual weather) will affect the urban transportation system.

<p align="center">
<img src=figures/1.png />
</p>


## Requirements

Our code is built upon [Latent Diffusion](https://github.com/CompVis/latent-diffusion).
```
git clone https://github.com/ChyaZhang/ChatTraffic.git
cd ChatTraffic
conda env create -f environment.yaml
conda activate ChatTraffic
```
## Data Preparation

For the datasets download please refer to [BjTT: A Large-scale Multimodal Dataset for Traffic Prediction](https://chyazhang.github.io/BjTT).

After getting the datasets, put them under a directory as follows:
```
ChatTraffic
    datasets/
        traffic/
            train/
                data/
                    1_1.npy
                    1_2.npy
                    ...
                text/
                    1_1.txt
                    1_2.txt
            validation/
                data/
                    1_6697.npy
                    1_6698.npy
                    ...
                text/
                    1_6697.txt
                    1_6698.txt
            matrix.npy
            Roads1260.json
            train.txt
            validation.txt
```
## Model Training

### Training autoencoder model
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/autoencoder/autoencoder_traffic.yaml -t --gpus 0,  
```
### Training diffusion model
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/traffic.yaml -t --gpus 0,
```

## Generation
```
python scripts/chattraffic.py --prompt "January 19, 2022, 15:48. road closure on south second ring road. a general traffic accident on s50 east fifth ring road. a general traffic accident on sihui bridge. road closure on wufang bridge. ......"
```

The output samples are in `.npy` format, you can use `scripts/plot_map.py` to visualize the traffic data on the map.




## Acknowledgments

Our code borrows heavily from [Latent Diffusion](https://github.com/CompVis/latent-diffusion).

## BibTeX

If you find this work useful for you, please cite
```
@ARTICLE{10803279,
  author={Zhang, Chengyang and Zhang, Yong and Shao, Qitan and Li, Bo and Lv, Yisheng and Piao, Xinglin and Yin, Baocai},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={ChatTraffic: Text-to-Traffic Generation via Diffusion Model}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TITS.2024.3510402}}
```
