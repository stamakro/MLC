import numpy as np
from scipy.optimize import minimize

#only need for testing and calculating performance
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def mlcTrain(Xtrain, ytrain, a, w0=None, maxIter=30):

	[N, f] = Xtrain.shape
	ytrain = ytrain.reshape(-1,)

	assert ytrain.shape[0] == N

	if w0 is None:
		w0 = np.ones(f)



	return w0


def mlcPredict(Xtrain, ytrain, Xtest, w):


	pass


def pccPredict():



if __name__ == "__main__":
	print 'Begining demo...'
