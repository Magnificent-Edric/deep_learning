import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float_
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0
        #print(ix)
        x1 = x.copy()
        x2 = x.copy()
        x1[ix] = x1[ix] + delta
        x2[ix] = x2[ix] - delta
        numeric_grad_at_ix = (f(x1)[0] - f(x2)[0]) / (2* delta)
        # print(len(ix), ix)
        # for i in range(len(ix)):
        #     numeric_grad_at_ix = (f(x[ix[0]][i] + delta)[0] - f(x[ix[0]][i] - delta)[0]) / (2* delta)
        # print('----')
        #положим в каждый элементы матрицы смещение дельта в обе допустимые дельтой области, для работы с многоразмерностью, чтобы использвать сразу функцию расчета общего ответа
        # print(f(x))
        # element_of_space_delta_plus = x + delta
        # # print(element_of_space_delta_plus)
        # element_of_space_delta_minus = x - delta
        # print((f(element_of_space_delta_plus)[0] - f(element_of_space_delta_minus)[0]) / (2 * delta))
        # numeric_grad_at_ix = (f(element_of_space_delta_plus)[0] - f(element_of_space_delta_minus)[0]) / (2 * delta)
        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
