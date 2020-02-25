import pandas as pd
import pickle
import numpy as np
import lifelines
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from collections import Counter
import sys
import json

sys.path.append('/Users/matthewbernstein/Development/single-cell-hackathon/src/common')

import utils
import load_GSE103224_TCGA_GBM

N_FOLDS = 5

def main():
    train_df = pd.read_csv('train_TCGA_GBM.tsv', sep='\t')
    train_df = train_df.set_index('sample_id')

    with open('cross_validation_null_distr.json', 'r') as f:
        null_samples = json.load(f)

    np.random.seed(8)
    shuff_inds = np.arange(len(train_df))
    np.random.shuffle(shuff_inds)
    folds = np.array_split(shuff_inds, N_FOLDS)
    print(folds)

    # Get genes used in 
    de_f = '../single-cell-hackathon/tmp/all_tumors_cluster_top_de_genes_aligned.json'
    with open(de_f, 'r') as f:
        clust_to_de = json.load(f)
    all_de_genes = set()
    for clust, de_genes in clust_to_de.items():
        all_de_genes.update(de_genes)
    keep_genes = sorted(all_de_genes)
    gene_name_to_index = {
        gene: index
        for index, gene in enumerate(
            load_GSE103224_TCGA_GBM.GENE_NAMES
        )
    }
    keep_gene_inds = [
        gene_name_to_index[gene]
        for gene in keep_genes
    ]

    Y = train_df[['time','censor']]

    print('Loading data...')
    X = load_GSE103224_TCGA_GBM.counts_matrix_for_sample_ids(Y.index)
    X = utils.log_cpm(X)
    restrict = False
    if restrict:
        X = X[:,keep_gene_inds]
    else:
        keep_genes = load_GSE103224_TCGA_GBM.GENE_NAMES
    print('done.')

    X = pd.DataFrame(
        data=X,
        index=Y.index,
        columns=keep_genes
    )

    if not restrict:
        stds = X.std()
        thresh = np.quantile(stds, 0.8)
        print('Removing all genes with st. dev < {}'.format(thresh))
        X = X.drop(X.std()[X.std() < thresh].index.values, axis=1)
        print(X.shape)
        X = X.loc[:,~X.columns.duplicated()]
    print('Shape of X: ', X.shape)
    df = Y.join(X)

    #penalties = [0.1, 1.0, 10.0, 100.0, 10e3, 10e4]
    penalties = [10.0, 100.0, 10e3, 10e4, 10e5, 10e6]
    alpha = 0.0
    means = []
    for lam in penalties:
        c_indices = utils.cross_fold_validation(df, folds, alpha, [lam])
        means.append(np.mean(c_indices))
    print(means)

    pvals = utils.empirical_pvalue(means, null_samples)
    print(pvals)

if __name__ == "__main__":
    main()
