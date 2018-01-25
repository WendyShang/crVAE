# Channel-Recurrent VAE for Image Modeling [[pdf](https://arxiv.org/pdf/1706.03729.pdf)]
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
  -To train Birds with baseline VAE-GAN, 
```bash
th main.lua -data /path/to/Birds/ -save /path/to/checkpoints/ -alpha 0.0002 -beta 0.05 -LR 0.0003 -eps 1e-6 -mom 0.9 -step 60 -manualSeed 1196
``` 
  -To train Birds with channel-recurrent VAE-GAN,
```bash
th main.lua -data /path/to/Birds/ -save /path/to/checkpoints/ -alpha1 0.0003 -alpha2 0.0002 -beta 0.0125 -LR 0.0003 -kappa 0.02 -latentType lstm -eps 1e-6 -mom 0.9 -step 60 -manualSeed 96
```
  -To train MNIST with VAE, 
```bash
th main_mnist.lua -LR 0.0003 -alpha 0.001 -latentType baseline -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binary/mnist/
```
  -To train MNIST with convolutional VAE, 
```bash
th main_mnist.lua -LR 0.0003 -alpha 0.001 -latentType conv -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binary/mnist/
```
  -To train MNIST with channel-recurrent VAE,
```bash
th main_mnist.lua -LR 0.003 -timeStep 8 -alpha 0.001 -latentType lstm -dataset mnist_28x28 -baseChannels 32 -nEpochs 200 -eps 1e-5 -mom 0.1 -step 50 -save /path/to/save/ -dynamicMNIST /path/to/dynamics/mnist/ -binaryMNIST /path/to/binayr/mnist/
```

## Pretrained Models (Coming Soon!)


## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{shang2017channel,
  title={Channel-Recurrent Autoencodering for Image Modeling},
  author={Shang, Wenling and Sohn, Kihyuk and Tian, Yuandong},
  booktitle={WACV},
  year={2018}
}
```
If you use the Birds data, please also cite the following papers
```
@article{wah2011caltech,
  title={The caltech-ucsd birds-200-2011 dataset},
  author={Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge},
  year={2011},
  publisher={California Institute of Technology}
}
@inproceedings{van2015building,
  title={Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection},
  author={Van Horn, Grant and Branson, Steve and Farrell, Ryan and Haber, Scott and Barry, Jessie and Ipeirotis, Panos and Perona, Pietro and Belongie, Serge},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={595--604},
  year={2015}
}
@inproceedings{berg2014birdsnap,
  title={Birdsnap: Large-scale fine-grained visual categorization of birds},
  author={Berg, Thomas and Liu, Jiongxin and Woo Lee, Seung and Alexander, Michelle L and Jacobs, David W and Belhumeur, Peter N},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2011--2018},
  year={2014}
}
```
If you use the dynamic MNIST dataset, please also cite
```
@article{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann},
  journal={http://yann. lecun. com/exdb/mnist/}
}
```
If you use the static MNIST datset, please also cite
```
@article{uria2016neural,
  title={Neural autoregressive distribution estimation},
  author={Uria, Benigno and C{\^o}t{\'e}, Marc-Alexandre and Gregor, Karol and Murray, Iain and Larochelle, Hugo},
  journal={Journal of Machine Learning Research},
  volume={17},
  number={205},
  pages={1--37},
  year={2016}
}
```

## Acknowledgments
