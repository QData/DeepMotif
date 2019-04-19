# Deep Motif Dashboard: Visualizing and Understanding Genomic Sequences Using Deep Neural Networks
### Jack Lanchantin, Ritambhara Singh, Beilun Wang, and Yanjun Qi
### Pacific Symposium on Biocomputing (PSB) 2017
https://arxiv.org/abs/1608.03644

### Talk slides:
https://github.com/QData/DeepMotif/blob/master/psb_talk_slides.pdf

### bibtex:
```
@inproceedings{lanchantin2017deep,
  title={Deep motif dashboard: Visualizing and understanding genomic sequences using deep neural networks},
  author={Lanchantin, Jack and Singh, Ritambhara and Wang, Beilun and Qi, Yanjun},
  booktitle={PACIFIC SYMPOSIUM ON BIOCOMPUTING 2017},
  pages={254--265},
  year={2017},
  organization={World Scientific}
}
```

[![LICENSE](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/QData/DeepMotif/blob/master/LICENSE)


# Installation


## Lua setup
The main modeling code is written in Lua using [torch](http://torch.ch)
Installation instructions are located [here](http://torch.ch/docs/getting-started.html#_)

After installing torch, install / update these packages by running the following:

```bash
luarocks install torch
luarocks install nn
luarocks install optim
```

### CUDA support (Optional)
To enable GPU acceleration with [CUDA](https://developer.nvidia.com/cuda-downloads), you'll need to install CUDA 6.5 or higher as well as [cutorch](https://github.com/torch/cutorch) and [cunn](https://github.com/torch/cunn). You can install / update the torch CUDA libraries by running:

```bash
luarocks install cutorch
luarocks install cunn
```

## LFS

Install git large file storage (LFS) in order to download the dataset directly from this git repository.

https://git-lfs.github.com/


## Visualization Method Dependencies

Weblogo: http://weblogo.berkeley.edu/

R: https://www.r-project.org/


# Usage


## Step 1: Get the Data
tar xvzf data/deepbind.tar.gz -C data/


## Step 2: Train the model
You can train one of the 3 types of models (CNN, RNN, or CNN-RNN). Check the flags in main.lua for parameters to run the code.

For CNN model:
```bash
th main.lua -cnn
```

For CNN model:
```bash
th main.lua -rnn
```

For CNN-RNN model:
```bash
th main.lua -cnn -rnn
```

## Step 3: Visualize the Model's Predictions
Once you have trained models, you can visualize the predictions. 


Saliency Map
```bash
th saliency_map.lua
```

Temporal Output Values
```bash
th temporal_output_values.lua
```

Class Optimization
```bash
th class_optimization.lua
```



