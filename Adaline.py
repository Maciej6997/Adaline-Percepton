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
    






   
    """ 
    Class Attributes :
    ---------
    eta : float
        Rate learning between 0 and 1
    n : int
        Iteraton number on train dataset
    random_state : int
        seed of random numbers used to initialize random weights
    shuffle : bool (default = True)
            If is True then shuffle train dataset before each epoch 
            to avoid cyclicality

    Dynamical Attributes :
    ----------
    w_ : one-dimensional array
        weights after fitting
    cost_ : list 
        Sum squares erros in each epoch
    
    """

    def __init__(self, eta = 0.01, n = 50, random_state = 42, shuffle = True):
        self.eta = eta
        self.n = n
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_initialized = False
        

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

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            cost = []

            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    

    def partial_fir(self, X, y):
        """
        Fitting learning data without re-initiation weights
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X, y)
        return self
    

    def _shuffle(self, X, y):
        r =self.rgen.permutation(len(y))
        return X[r], y[r]
        
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 0.01, size = 1 + m)
        self.w_initialized = True



    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost



    def net_input(self, X):
        """
        Calculate cumulate boost
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    

    def activation(self, X):
        """Compute activate function"""
        return X
        

    def predict(self, X):
        """     
        Return class after calculate the unit stroke function
        """

        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)