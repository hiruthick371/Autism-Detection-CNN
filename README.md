
# VGG16 Fine-Tuning with LoRA

## Overview
This project implements fine-tuning of the VGG16 model using Low-Rank Adaptation (LoRA) for Autism Detection. LoRA is a technique that allows efficient adaptation of pre-trained models with fewer trainable parameters, making it suitable for various tasks such as image classification and etc.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites
- Python 3.x
- PyTorch
- torchvision
- other libraries

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/karpagam-final-year-project/autism-detection.git
cd autism-detection
```

## Dataset
You can download the dataset from the following link:

- [Dataset Link](https://www.kaggle.com/datasets/cihan063/autism-image-data)

## Usage
To fine-tune the VGG16 model using LoRA for Autism Detection, run the training script:

```bash
python train.py
```
## Training
During training, the model will save checkpoints and logs. You can monitor the training progress by looking at the logs in the `logs` directory.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [VGG16](https://arxiv.org/abs/1409.1556) for the original architecture.
- [LoRA](https://arxiv.org/abs/2106.09685) for the adaptation technique.