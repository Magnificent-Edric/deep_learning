import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax

def execute_forward(data, current_layer):
    return current_layer.forward(data)

def execute_backward(data, current_layer):
    return current_layer.backward(data)

class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.hidden_size = hidden_layer_size
        # TODO Create necessary layers
        self.sequence_of_layers = []
        self.sequence_of_layers.append(FullyConnectedLayer(n_input, self.hidden_size))
        self.sequence_of_layers.append(ReLULayer())
        self.sequence_of_layers.append(FullyConnectedLayer(self.hidden_size, n_output))
        # raise Exception("Not implemented!")

    def return_params(self):
        return {'hidden_size':self.hidden_size, 'reg': self.reg}

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        
        grad = np.zeros(y.shape[0])      
        predictions = X
        #Forward pass through the model
        for layer in self.sequence_of_layers:
            predictions = execute_forward(predictions, layer)
        loss, grad = softmax_with_cross_entropy(predictions, y)

        l2_loss = 0
        #Backward pass throught the model
        for i in reversed((self.sequence_of_layers)):
            grad = execute_backward(grad, i)
        #сверху производна по элементам слоев входящих в вычисление предсказываемого
        #внизу с регуляризации разложение  на две составляющие
        for i in reversed((self.sequence_of_layers)):
            for param in i.params():
                param = layer.params()[param]
                # print(param.grad)
                l2, grad_l2 = l2_regularization(param.value, self.reg)
                # print(param.grad.shape)
                param.grad += grad_l2
                l2_loss += l2
                # grad_l2 = np.zeros(len(grad_l2))
                # print(param.grad[0][0], grad[0][0] + grad_l2[0][0])
                # print("----")
                # print(grad.shape, grad_l2.shape)
        loss += l2_loss
        
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int_)
        pred = np.array(pred)
        predictions = X
        #Forward pass through the model
        for layer in self.sequence_of_layers:
            predictions = execute_forward(predictions, layer)
        predictions = softmax(predictions)
        pred = np.argmax(predictions, axis=1)
        # return np.argmax(pred)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        # print(self.sequence_of_layers[0].params()['W'].value)
        for i in range(len(self.sequence_of_layers)):
            for param in self.sequence_of_layers[i].params():
                result[str(i) + "_" + param] = self.sequence_of_layers[i].params()[param]
        return result
