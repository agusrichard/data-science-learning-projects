import numpy as np
import pandas as pd

def train_test_split(X, y, random_state=0, test_size=0.2):
    """
    Split the data into train set and test set
    
    ...
    
    Parameters
    ----------
    X : array-like object
        Feature matrix
    y : array-like object
        Target vector
    random_state : number, optional
        Seed for random number generator (default=0)
    test_size : float between 0 and 1
        The ratio of test set size (default=0.2)
        
    """
    
    
    rng = np.random.RandomState(seed=random_state)
    test_size = int(len(X) * test_size)
    permutation = rng.permutation(len(X))
    permutated_test = permutation[:test_size]
    permutated_train = permutation[test_size:]
    return X[permutated_train], X[permutated_test], y[permutated_train], y[permutated_test]
    
        
        
    