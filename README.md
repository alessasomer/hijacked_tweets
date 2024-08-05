# Detecting Hijacked Tweets Using a Deep Averaging Network (DAN)

This project explores the detection of hijacked tweets on Twitter using a Deep Averaging Network (DAN) for my final project in NLP (CS 375). The study focuses on the hashtags #MeToo and #coronavirus, aiming to classify tweets as either hijacked or non-hijacked.

## Project Overview

- **Objective**: To classify tweets as hijacked or non-hijacked using a DAN model.
- **Data Sources**: 
  - **#MeToo Tweets**: A dataset from Mousavi and Ouyang (2021).
  - **#coronavirus Tweets**: A manually curated dataset collected in May 2023.
- **Tech Stack**: Python, Pytorch, GloVe Embeddings, Tweepy, NLTK

## Introduction

Hashtag hijacking occurs when users misuse a popular hashtag, either to disrupt communication or to promote unrelated content. This project aims to develop a machine learning model capable of detecting such hijacked tweets, which is crucial for preserving the integrity of social movements and public health information on platforms like Twitter.

## Methodology

1. **Data Collection and Preprocessing**:
   - **Data Collection**: Used the Twitter Developer API to collect tweets and the Tweepy package for easier interaction. Tweet data was then tokenized using NLTK's `TweetTokenizer`.
   - **GloVe Embeddings**: Employed GloVe Twitter-specific embeddings for vector representation of words.
   - **Data Storage**: Preprocessed data was serialized using Pickle for efficient storage and retrieval.

2. **Model Construction**:
   - **Deep Averaging Network (DAN)**: A simple yet effective neural network architecture was used. The model includes two hidden layers and utilizes Leaky ReLU as the activation function.
   - **Training and Evaluation**: The model was trained on the #MeToo dataset and evaluated on both #MeToo and #coronavirus datasets. A Grid Search was conducted to optimize hyperparameters such as learning rate, dropout probability, and negative slope.

3. **Grid Search for Hyperparameter Tuning**:
   - The Grid Search explored various combinations of learning rates, dropout probabilities, and negative slopes to identify the best parameters.

## Read the full paper here!
https://drive.google.com/file/d/1vtxOAkKRs68TnIKO3wmusHDg-oXMpKIH/view
