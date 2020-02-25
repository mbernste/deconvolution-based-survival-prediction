import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import numpy as np
import lifelines
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from collections import Counter
from lifelines.plotting import qq_plot
from glmnet_py import glmnet, glmnetPredict; from glmnetPlot import glmnetPlot
import json

import utils

N_FOLDS = 5


def main():
    
    train_df = pd.read_csv('train_TCGA_GBM.tsv', sep='\t')
    train_df = train_df.set_index('sample_id')
    decon_df = pd.read_csv('deconvolved_TCGA_GBM.tsv', sep='\t', index_col=0)
    
    with open('cross_validation_null_distr.json', 'r') as f:
        null_samples = json.load(f)

    Y = train_df[['time','censor']]
    X = decon_df.loc[Y.index]

    np.random.seed(8)
    shuff_inds = np.arange(len(Y))
    np.random.shuffle(shuff_inds)
    folds = np.array_split(shuff_inds, N_FOLDS)
        
    cols = X.columns
    col_to_new_col = {
        col: 'Clust_{}'.format(col) 
        for col in cols
    }
    X.rename(
        columns=col_to_new_col,
        inplace=True
    )
    df = Y.join(X)

    lambdas = [0.01, 0.1, 10.0, 100.0, 10e3, 10e4]
    alpha = 0.0
    means = []
    for lam in lambdas:
        c_indices = utils.cross_fold_validation(df, folds, alpha, [lam])
        print(c_indices)
        means.append(np.mean(c_indices))
    print(means)

    pvals = utils.empirical_pvalue(means, null_samples)
    print(pvals)



if __name__ == "__main__":
    main()
