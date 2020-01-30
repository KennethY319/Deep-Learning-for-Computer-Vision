import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    CorrectScore = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - CorrectScore + 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i, :]
        dW[:, j] += X[i, :]

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape)
  scores = X.dot(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]

  CorrectScore = scores[range(num_train), list(y)].reshape(-1,1)
  margins = np.maximum(0, scores-CorrectScore+1)
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins) / num_train+0.5 * reg * np.sum(W * W)

  margins[margins > 0] = 1
  row_sum = np.sum(margins, axis=1)
  margins[np.arange(num_train), y] = -row_sum
  dW += np.dot(X.T, margins) / num_train + reg * W

  return loss, dW
