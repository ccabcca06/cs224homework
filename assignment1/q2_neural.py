#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    # YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(X, W1) + b1)
    # 从输入X到隐藏层h的映射
    y_hat = softmax(np.dot(h, W2) + b2)
    # 从隐藏层h到输出y_hat的映射
    # END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    cost = np.sum(-np.log(y_hat[labels == 1])) / X.shape[0]
    d3 = (y_hat - labels) / X.shape[0]
    # 按照交叉熵损失函数的定义：CE(y,y_hat)=sum（y*log(y_hat))构建损失函数

    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3, 0, keepdims=True)

    dh = np.dot(d3, W2.T)
    grad_h = sigmoid_grad(h) * dh

    gradW1 = np.dot(X.T, grad_h)
    gradb1 = np.sum(grad_h, 0)

    # 反向传播梯度误差回传，通过矩阵乘法实现
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    # np.concatenate用于拼接向量，flatten()将多维数组"压扁"到一维（即转换为向量）
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
