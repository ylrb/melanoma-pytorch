# SIIM-ISIC Melanoma Classification

This repository contains code for a PyTorch-based melanoma classification model using EfficientNet architecture, achieving a Kaggle score of **0.88** on the [SIIM-ISIC Melanoma Classification challenge](https://www.kaggle.com/c/siim-isic-melanoma-classification/).

## Overview

The task is to classify skin lesion images as benign (0) or malignant (1) using the provided dataset. 

The model employs EfficientNet-B0 architecture for feature extraction and classification, using AdamW optimizer with weight decay and cross-entropy loss.
Random flips and data augmentation from 2019 & 2020 datasets helps countering the overfitting caused by class inbalance.

## Usage

Simply run `EfficientNet.py` to train the model with the correct file paths and hyperparameters
