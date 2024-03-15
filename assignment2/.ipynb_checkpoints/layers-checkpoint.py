import numpy as np

def l2_regularization(W, reg):
    # grad = W.copy()
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    loss = 0.5 * reg * np.sum(W**2)
    grad = np.dot(reg, W)
    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        probs = predictions.copy()
        probs -= np.max(probs)
        return np.exp(probs) / np.sum(np.exp(probs))

    probs = predictions.copy()
    probs -= np.max(probs, axis=1).reshape(-1, 1)
    softmax = np.exp(probs) / np.sum(np.exp(probs), axis=1).reshape(-1, 1)
    return softmax


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    H = 0.0
    # TODO implement cross-entropy
    if len(probs.shape) == 1: 
        H = -np.sum(np.log(probs[target_index]))
        return (H)
    # print("-----")
    # print(probs.shape[0])
    # print("-----")
    H = -np.sum(np.log(probs[np.arange(probs.shape[0]), target_index])) # (batch_size, N)
    # Your final implementation shouldn't have any loops
    # print("-----")
    # print((H ))
    # print("-----")
    # H /= probs.shape[0] #усредняем ошибку по батчу
    return H


def softmax_with_cross_entropy(predictions, target_index):
        '''
        Computes softmax and cross-entropy loss for model predictions,
        including the gradient
    
        Arguments:
          predictions, np array, shape is either (N) or (batch_size, N) -
            classifier output
          target_index: np array of int, shape is (1) or (batch_size) -
            index of the true class for given sample(s)
    
        Returns:
          loss, single value - cross-entropy loss
          dprediction, np array same shape as predictions - gradient of predictions by loss value
        '''
        # TODO implement softmax with cross-entropy
        # Your final implementation shouldn't have any loops
        probs = softmax(predictions)
        loss = cross_entropy_loss(probs, target_index)
        dprediction = probs.copy()
        #it's naive implementation of softmax classifier with cross entropy loss (векторизированя реализация лучше)
        if len(predictions.shape) == 1:
            dprediction[target_index] -= 1
            return loss, dprediction
            
        dprediction[np.arange(dprediction.shape[0]), target_index] -= 1
    
        return loss, dprediction

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.Xrelu = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.Xrelu = X
        return np.maximum(0.0, X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        d_result = d_out.copy()
        d_result[self.Xrelu < 0] = 0 #тут идея в тотм, что производная релу = 1, но мы умножаем на предыдущее значение(сложная функция), а при отрицателььны хзначения 0
        # print(d_result[self.Xrelu > 0])
        # print(self.Xrelu > 0)
        # print(self.Xrelu)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        prediction = X.dot(self.W.value) + self.B.value
        return prediction

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = d_out.copy()
        self.W.grad = ((self.X).T).dot(d_out)
        d_input = np.dot(d_out, self.W.value.T)
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
