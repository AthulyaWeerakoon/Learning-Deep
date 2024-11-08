# Deep Learning Project

Welcome to the **Deep Learning Project** repository! This project serves as a personal learning journey in deep learning, exploring fundamental techniques, neural network architectures, and experiments primarily using Python, Numpy, Pandas, and TensorFlow.

## Table of Contents
- [Sigmoid Neural Network](#sigmoid-nn)
- [Regression with Keras](#regression-with-keras)
- [Classification with Keras](#classification-with-keras)
- [Classification using CNN with Keras](#classification-using-cnn-with-keras)
- [Actor critic reinforcement with Keras](#actor-critic-reinforcement-with-keras)
- [Simple image caption Generator](#simple-image-caption-generator)

## Introduction

This project repository is designed to organize and track the development of various deep learning exercises and experiments. It's an open-ended project aimed at mastering essential concepts in machine learning and deep learning using practical implementations and examples.

### Sigmoid NN
[This](https://github.com/AthulyaWeerakoon/Learning-Deep/blob/main/sigmoid_neural_network.py) is a neural network with sigmoid activation function written from scratch with forward and backward propagation. Suffers greatly from vanishing gradient problem. 

### Regression with Keras
[This](https://github.com/AthulyaWeerakoon/Learning-Deep/blob/main/Regression_with_keras.py) is a simple python program that attempts linear regression.

### Classification with Keras
[This](https://github.com/AthulyaWeerakoon/Learning-Deep/blob/main/Classification_with_keras.py) is a simple python program that classifies using the mnist handwritten number dataset

### Classification Using CNN with Keras
[This](https://github.com/AthulyaWeerakoon/Learning-Deep/blob/main/CNNs_with_keras.py) is the same as the earlier program, but instead of using a dense layer of size 28x28 it uses convolutional and pooling layers.

### Actor critic reinforcement with Keras
[This](https://github.com/AthulyaWeerakoon/Learning-Deep/blob/main/Actor_critic_reinforcement.py) is an incomplete implementation of a deep actor critic reinforcement learning model attempting to face the cart-pole problem. Required opengym library to run the simulations.

### Simple image caption generator
[This](https://github.com/AthulyaWeerakoon/Learning-Deep/blob/main/Simple%20Image%20Caption%20Gen/ImageCaptionGenerator.ipynb) is a simple image caption generator model (a show and tell model) with encoder and decoder models utilizing a pretrained ResNET feature extractor. Tensorflow ArtificialDataset is used to pipeline the training process and is set up to be hyperparametrically tuned using a wandb sweep. Project report can be accessed [here](https://www.linkedin.com/in/athulya-weerakoon/details/projects/) in this linked-in profile (look under the media files for the image caption generator project).
