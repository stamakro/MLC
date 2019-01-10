import numpy as np
from scipy.optimize import minimize

#only need for testing and calculating performance
from sklearn.model_selection import KFold
8from sklearn.metrics import roc_auc_score

def tvalueG(w, X, y, a):

		

		#determine genes with and without GO term (positive and negative genes)
        pos = np.where(y)[0]
        neg = np.where(y == 0)[0]

		#subsample the number of genes to reduce computation to at most 100 positives and 100 negatives
        np.random.seed(2312)
        maxnr = 100
        pos = np.random.permutation(pos)[:maxnr]
        neg = np.random.permutation(neg)[:maxnr]

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
#
