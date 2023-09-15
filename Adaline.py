#import bibliotek 

import numpy as np 

class AdalineGradientDescent:
    """ 
    
    Class Attributes :
    ---------
    eta : float
        Rate learning between 0 and 1
    n : int
        Iteraton number on train dataset
    random_state : int
        seed of random numbers used to initialize random weights
    
    Dynamical Attributes :
    ----------
    w_ : one-dimensional array
        weights after fitting
    cost_ : list 
        Sum squares erros in each epoch
    
    """

    def __init__(self, eta = 0.01, n = 50, random_state = 42):
        self.eta = eta
        self.n = n
        self.random_state = random_state

    def fit(self, X, y):

        """
        Function fitting model to data

        Parameters: 
        -------

        X : Array(N x M)
            Array with learning vectors
            N - row number (observation number)
            M - column number (feature number) 
        
        y : Array (N x 1)
            Vector with targes values

        Returns: 
        ----------
        self : object
    
        """

        #create instance of random number generator
        rgen = np.random.RandomState(self.random_state)
        #Create a random weights vector from normal distributions
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []

        for k in range(self.n):
            net_input = self.net_input(X)
            output = self.activation_function(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            self.cost_.append((errors ** 2).sum() / 2.0)
        
        return self
        

    def net_input(self, X):
        """
        Calculate cumulate boost
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    

    def activation_function(self, X):
        """Compute activate function"""
        return X
        

    def predict(self, X):
        """     
        Return class after calculate the unit stroke function
        """

        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, -1)
    






   