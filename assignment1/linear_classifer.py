import numpy as np


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
    # f = predictions.copy()
    # f -= np.max(f)
    # softmax = np.exp(f) / np.sum(np.exp(f))
    
    #For batch_size
    if len(predictions.shape) == 1:
        probs = predictions.copy()
        probs -= np.max(probs)
        return np.exp(probs) / np.sum(np.exp(probs))

    probs = predictions.copy()
    probs -= np.max(probs, axis=1).reshape(-1, 1)
    softmax = np.exp(probs) / np.sum(np.exp(probs), axis=1).reshape(-1, 1)
    
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
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
    H = -np.sum(np.log(probs[np.arange(probs.shape[0]), target_index])) # (batch_size, N)
    # Your final implementation shouldn't have any loops
    H /= probs.shape[0] #усредняем ошибку по батчу
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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    grad = W.copy()
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    loss = 0.5 * reg_strength * np.sum(W**2)
    grad = np.dot(reg_strength, W)
    return loss, grad
    

def linear_softmax(X, W, target_index, pred, flag=False):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    if flag == False:
        predictions = np.dot(X, W)
    else:
        predictions = pred
    loss, dpred = softmax_with_cross_entropy(predictions, target_index)
    dW = (X.T).dot(dpred) / X.shape[0] #усредняем параметры по батчу != усрденению ошибки 
    #  TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    
    
    return loss, dW

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None
        self.lr = None
        self.reg = None
        
    def return_parmas(self):
        return self.lr, self.reg, self.W 
    def fit(self, X, y, batch_size, learning_rate, reg,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''
        self.lr = learning_rate
        self.reg = reg
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)
        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for batch in batches_indices:
                train_x = X[batch]
                train_y = y[batch]
                scores = train_x.dot(self.W)
                loss, dw = linear_softmax(train_x, self.W, train_y, scores, True)
                loss1, dw1 = l2_regularization(self.W, reg)
                loss = loss + loss1
                self.W += -learning_rate * (dw + dw1)
            # raise Exception("Not implemented!")
            loss_history.append(loss)
            # end
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int_)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")
        y_predd = X.dot(self.W)
        y_pred = np.argmax(y_predd, axis=1)
        return y_pred



                
                                                          

            

                
