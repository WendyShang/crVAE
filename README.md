# Channel-Recurrent VAE for Image Modeling [pdf](https://arxiv.org/pdf/1706.03729.pdf)
## Prerequisites
  - Linux, NVIDIA GPU + CUDA CuDNN 
  - Install torch dependencies from https://github.com/torch/distro
  - Install torch pacakge `cudnn`
```bash
luarocks install cudnn
```
  - Install the **batchDisc** branch of the git repo [stnbhwd](https://github.com/qassemoquab/stnbhwd/tree/batchDisc), since we will need the batch discrimination layer. 

## Dataset
  - We provide code with Birds dataset. You may download the processed t7 file from https://surfdrive.surf.nl/files/index.php/s/MeQvGwtRGf1W6e8.
  - We also provide ablation studies with MNIST. You may download the MNIST files (both binary and dynamic) from https://surfdrive.surf.nl/files/index.php/s/MeQvGwtRGf1W6e8.

## Training 
  -To train Birds with the baseline VAE-GAN, 
```bash
th main.lua -data /path/to/Birds/ -save /path/to/checkpoints/ -alpha 0.0002 -beta 0.05 -LR 0.0003 -eps 1e-6 -mom 0.9 -step 60 -manualSeed 1196
``` 
## Pretrained Models (Coming Soon!)
