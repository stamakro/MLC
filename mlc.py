import numpy as np
from scipy.optimize import minimize

#only needed for testing and calculating performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def tvalueG(w, X, y, a):
        '''

Calculates the current value of the MLC objective function as well as its parital derivatives with
respect to the weights vector w. It is called multiple times during the optimization process

Input Arguments:
w: an array of length #samples containing sample weights
X: a 2-d gene expression array (#genes x #samples)
y: a label vector of length #genes, y[i] == 1 iff gene i is annotated with the Go term/pathway in question and 0 otherwise
a: floating point, the regulatization parameter of the MLC objective

Returns:
A tuple.
The first element is a floating point containing the value of the cost function given the inputs
The second is an array of size #samples containing the partial derivatives of the cost function with respect
to the weights

        '''
	#determine genes with and without GO term (positive and negative genes)
        pos = np.where(y)[0]
        neg = np.where(y == 0)[0]


        N1 = pos.shape[0]
        N2 = neg.shape[0]

        Npp = N1 * (N1 - 1) / 2
        Npn = N1 * N2

        Xpos = X[pos]
        Xneg = X[neg]

        indices = np.triu_indices(N1, k=1)
        #all pairwise weighted inner products of positive gene pairs
        Spp = (Xpos * w).dot(Xpos.T)[indices].flatten()
        
		#all pairwise weighted inner products of positive-negative gene pairs
        Spn = (Xpos * w).dot(Xneg.T).flatten()
        
        #mean and variance of weighted inner products for t-test
        X1 = np.mean(Spp)
        X2 = np.mean(Spn)

        v1 = np.var(Spp, ddof=1)
        v2 = np.var(Spn, ddof=1)

        #calculate the t-value:
        '''
         t = \frac{X_1 - X_2}{\sqrt{\frac{\sigma_1^2}{N_1} + \frac{\sigma_2^2}{N_2}}}
        '''
        numerator = (X1 - X2)
        denominator = np.sqrt((v1 / Npp) + (v2 / Npn))

        t = numerator / denominator


        #calculate the partial derivatives of t with respect to the weights w

        gradient = np.zeros((X.shape[1],))

        dx1_dw = np.zeros((X.shape[1],))
        dv1_dw = np.zeros((X.shape[1],))

        dx2_dw = np.zeros((X.shape[1],))
        dv2_dw = np.zeros((X.shape[1],))

        for ii in xrange(pos.shape[0] - 1):
                for jj in xrange(ii+1, pos.shape[0]):
                        dx1_dw += Xpos[ii] * Xpos[jj]

        dx1_dw /= Npp


        counter = -1
        for ii in xrange(pos.shape[0] - 1):
                for jj in xrange(ii+1, pos.shape[0]):
                        counter += 1

                        dv1_dw += (Spp[counter] - X1) * (Xpos[ii]*Xpos[jj] - dx1_dw)

        dv1_dw /= (Npp - 1)

        for ii in xrange(pos.shape[0]):
                for jj in xrange(neg.shape[0]):
                        dx2_dw += Xpos[ii] * Xneg[jj]

        dx2_dw /= Npn


        counter = -1
        for ii in xrange(pos.shape[0]):
                for jj in xrange(neg.shape[0]):
                        counter += 1

                        dv2_dw += (Spn[counter] - X1) * (Xpos[ii]*Xneg[jj] - dx2_dw)


        dv2_dw /= (Npn - 1)


        part1 = (dx1_dw - dx2_dw) / denominator

        part2 = ((X1 - X2) / (denominator ** 3)) * ( (dv1_dw / Npp) + (dv2_dw / Npn) )

        gradient = -a *(part1 - part2) + 1 - a

        cost = -a * t + (1 - a) * np.sum(w)


        return cost, gradient




def mlcTrain(Xtrain, ytrain, a, w0=None, maxIter=30):
        '''

Calculates the current value of the MLC objective function as well as its parital derivatives with
respect to the weights vector w. It is called multiple times during the optimization process

Input Arguments:
Xtrain: a 2-d gene expression array (#genes x #samples)
ytrain: a label vector of length #genes, y[i] == 1 iff gene i is annotated with the Go term/pathway in question and 0 otherwise
a:      a floating point, the regulatization parameter of the MLC objective
w0:     an array of size #samples containing an initial estimate of the weights. If None (default), all weights are initialized to 1
maxIter:the maximum number of iterations to be performed by the numerical optimizer. Defaults to 30.

Returns:
An array of size #samples containing the learned sample weights

        '''
	[N, f] = Xtrain.shape
	ytrain = ytrain.reshape(-1,)

	assert ytrain.shape[0] == N

	if w0 is None:
		w0 = np.ones(f)

	#determine genes with and without GO term (positive and negative genes)
	pos = np.where(ytrain)[0]
	neg = np.where(ytrain == 0)[0]

	#subsample the number of genes to reduce computation to at most 100 positives and 100 negatives
	np.random.seed(2312)
	maxnr = 100
	pos = np.random.permutation(pos)[:maxnr]
	neg = np.random.permutation(neg)[:maxnr] 

	ii = np.hstack((pos, neg))
	ytrain = ytrain[ii]
	Xtrain = Xtrain[ii]

	#bounds for weights, constraint each w[i] to be non-negative
	bb = tuple([(0.0, np.inf) for i in xrange(Xtrain.shape[1])])

	#minimize cost function
	solution = minimize(tvalueG, w0, args=(Xtrain, ytrain, a), jac=True, bounds=bb, options={'maxiter': maxIter})

	w = solution.x

	return w


def mlcPredict(Xtrain, ytrain, Xtest, w, k):
	'''
k-NN classifier with the weighted inner product of MLC as a distance function

Input Arguments:
Xtrain: training set, a 2-d gene expression array (#genes_train x #samples)
ytrain: training labels, a label vector of length #genes_train, y[i] == 1 iff gene i is annotated with the Go term/pathway in question and 0 otherwise
Xtest:  test set, a 2-d gene expression array (#genes_test x #samples)
w:      an array of size #samples that contains the weights learned by MLC during training
k:      the number of nearest neighbors to consider

Returns:
An array of size #genes_test containing the posterior probability that each gene is associated with the label

	'''
	C = (Xtest * w).dot(Xtrain.T)
	knns = np.argsort(C, axis=1)[:, -k:]

	ypred = np.sum(ytrain[knns], axis=1) / float(k)

	return ypred


def pccPredict(Xtrain, ytrain, Xtest, k):
	'''
For comparison: k-NN classifier with pearson correlation as a distance function

Input Arguments:
Xtrain: training set, a 2-d gene expression array (#genes_train x #samples)
ytrain: training labels, a label vector of length #genes_train, y[i] == 1 iff gene i is annotated with the Go term/pathway in question and 0 otherwise
Xtest:  test set, a 2-d gene expression array (#genes_test x #samples)
k:      the number of nearest neighbors to consider

Returns:
An array of size #genes_test containing the posterior probability that each gene is associated with the label

	'''
	Ntrn = Xtrain.shape[0]
	Ntst = Xtest.shape[0]

	# a Ntst x Ntrn matrix containing the correlation of all test genes to all training genes
	C = np.corrcoef(Xtrain, Xtest)[Ntrn:][:, :Ntrn]

	knns = np.argsort(C, axis=1)[:, -k:]
	ypred = np.sum(ytrain[knns], axis=1) / float(k)


	return ypred



if __name__ == "__main__":
	print('Generating random data...')
	#generate some random expression data, with 1000 genes and 30 samples
	X = np.random.rand(1000, 30)

	#generate random labels 0-1
	y = np.random.randint(0, 2, 1000)

	#split into training and test sets
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.333)

	print('Training MLC...') 
	#learn weights in training set, setting a to 0.9
	w = mlcTrain(Xtrain, ytrain, 0.9, w0=None, maxIter=30)

	#set the number of nearest neighbors
	k = 20

	print('Predicting with MLC...')
	#use learned weights to make predictions in the test set
	ypred_mlc = mlcPredict(Xtrain, ytrain, Xtest, w, k)

	print('Predicting with Pearson correlation...')
	#make prediction using pearson correlation, for comparison
	ypred_pcc = pccPredict(Xtrain, ytrain, Xtest, k)


	#evaluate the two predictors using roc auc
	auc_mlc = roc_auc_score(ytest, ypred_mlc)
	auc_pcc = roc_auc_score(ytest, ypred_pcc)

	print ('Performance of PCC:', auc_pcc)
	print ('Performance of MLC:', auc_mlc)



