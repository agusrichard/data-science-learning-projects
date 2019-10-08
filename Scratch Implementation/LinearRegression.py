from sklearn.base import BaseEstimator
from scipy.linalg import inv
from sklearn.linear_model import LinearRegression
import numpy as np

class LinearRegressionVersion1(BaseEstimator):

	"""Imitation of LinearRegression estimator from sklearn.
	It doesn't take any parameters or hyperparameters. 
	This estimator will use full-batch, which means that it uses
	all the data points and makes it slow to process a lot of data 
	points.

	Parameters: 
	----------

	Attributes:
	----------
	
	coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,\
            n_features)
        Weights assigned to the features.

    intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

	"""

	def __init__(self):
		pass
  		

	def fit(self, X, y):	

		X = np.c_[np.ones(X.shape[0]), X]
		weights = inv(X.T.dot(X)).dot(X.T).dot(y)
		self.coef_ = weights[1:]
		self.intercept_ = weights[0]
		return self

	def predict(self, X):

		X = np.c_[np.ones(X.shape[0]), X]
		return X.dot(np.hstack([self.coef_, self.intercept_]))


class LinearRegressionVersion2(BaseEstimator):

	"""This estimator will take sample from whole data points.
	Train and get the weights.

	Parameters: 
	----------
	
	sample_size : float (0-1), default=0.05
		Specify the percentage of data points in each iteration

	n_iter : integer, default=500
		The number of iterations

	replace : boolean, default=True
		Using boostrap if True, False otherwise

	random_state : number
		Seed for random number generator
	
	Attributes:
	----------
	
	coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,\
            n_features)
        Weights assigned to the features.

    intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    """

	def __init__(self, sample_size=0.05, n_iter=500, replace=True, random_state=42):

		self.sample_size = sample_size
		self.n_iter = n_iter
		self.replace = replace

	def fit(self, X, y):

		coefs = np.empty(X.shape[1])
		intercepts = np.empty(self.n_iter)
		for i in range(self.n_iter):
			choice = np.random.choice(X.shape[0], int(self.sample_size*X.shape[0]), self.replace)
			X_sampled = X[choice]
			y_sampled  = y[choice]
			model = LinearRegression().fit(X_sampled, y_sampled)
			coefs = np.vstack([coefs, model.coef_])
			intercepts[i] = model.intercept_
		self.coef_ = np.mean(coefs, axis=0)
		self.intercept_ = np.mean(intercepts)
		return self

	def predict(self, X):

		return X.dot(self.coef_) + self.intercept_



