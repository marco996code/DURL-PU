# Distributed Unsupervised Representation Learning for Remote Sensing Image Classification

This repository is the code of the paper **"**Distributed Unsupervised Representation Learning for Remote Sensing Image Classification**"**.

## Dataset

The custom remote sensing dataset named FAIR-M.

*   &#x20;**Download link**: <https://pan.baidu.com/s/1HvpBlvwlLlo4HB0hqjpFcA>
*   &#x20;**Extraction code**: `8am4`

go to the get\_dataset function in src/utils.py and modify the dataset path to your local directory.

## Training

To train the model , run:

python main.py

## Non-IID Setting

python main.py --dirichlet --dir\_beta 0.5

Run the following for evaluation:

python linear\_evaluation.py

