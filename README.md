# Distributed Unsupervised Representation Learning for Remote Sensing Image Classification

This repository is the code of the paper **"**Distributed Unsupervised Representation Learning for Remote Sensing Image Classification**"**.

## Dataset

The processed remote sensing dataset fair2m.

*   &#x20;**Download link**: <https://pan.baidu.com/s/1hJ0AIWz5DSoCjRFTLFZpug>
*   &#x20;**Extraction code**: `4ed3 `

go to the get\_dataset function in src/utils.py and modify the dataset path to your local directory.

## Training

To train the model , run:

python main.py

## Non-IID Setting

python main.py --dirichlet --dir\_beta 0.5

Run the following for evaluation:

python linear\_evaluation.py
