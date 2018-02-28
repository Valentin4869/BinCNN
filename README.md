# BinCNN

Binarized convolutional neural network implementation for vehicle classification in CUDA, timed against cuDNN+cuBLAS (7.5X speed up for cuDNN with GEMM convolution). Paper under review. The original non-binarized network architecture is according to https://arxiv.org/pdf/1602.07125.pdf

For more information about binarized neural networks:
http://papers.nips.cc/paper/6573-binarized-neural-networks.pdf
http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-binary-weights-during-propagations.pdf
