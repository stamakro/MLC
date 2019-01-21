# MLC
Metric Learning for Co-expression (MLC) is a gene function prediction method.
Starting from a set of genes that have a particular function as well as
a gene expression dataset, it assigns a weight value to each sample, so that
the weighted inner product similarity is maximized for genes with that function.

## Usage
The file *mlc.py* contains all the necessary code to run MLC
1. Use the *mlcTrain* function to learn weights from a set of annotated genes
2. Use the *mlcPredict* function to make predictions for a set of genes of unknown function

Type "python mlc.py" to execute a demo with random data.

## Dependencies
* numpy
* scipy
* scikit-learn (only for demo purposes, not used by MLC itself)

**Note:** The package has been tested using Python 2.7, numpy 1.14.3 and scipy 1.1.0