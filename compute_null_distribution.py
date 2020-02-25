import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
import lifelines
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from collections import Counter
import json

import utils

N_FOLDS = 5
N_SAMPLES = 1000

def main():
    train_df = pd.read_csv('train_TCGA_GBM.tsv', sep='\t')
    train_df = train_df.set_index('sample_id')
    decon_df = pd.read_csv('deconvolved_TCGA_GBM.tsv', sep='\t', index_col=0)

    Y = train_df[['time','censor']]
    X = decon_df.loc[Y.index]

    np.random.seed(8)
    shuff_inds = np.arange(len(Y))
    np.random.shuffle(shuff_inds)
    folds = np.array_split(shuff_inds, N_FOLDS)

    samples = []
    sample_ids = np.array(Y.index)
    all_means = []
    for sample_i in range(N_SAMPLES):
        if (sample_i + 1) % 10 == 0:
            print('Generated {} samples...'.format(sample_i+1))
        c_indices = []
        for fold_i in range(N_FOLDS):
            fold_size = int(len(Y)/N_FOLDS)
            rand_scores = np.random.rand(fold_size)
            Y_samp = Y.sample(fold_size)
            c_index = concordance_index(
                Y_samp['time'],
                rand_scores,
                event_observed=Y_samp['censor']
            )
            c_indices.append(c_index)
        all_means.append(np.mean(c_indices))
    with open('cross_validation_null_distr.json', 'w') as f:
        json.dump(list(all_means), f, indent=4)
    sns.distplot(all_means, hist=False)
    plt.show()



if __name__ == "__main__":
    main()
