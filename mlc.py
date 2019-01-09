import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
