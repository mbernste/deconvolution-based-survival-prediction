import numpy as np
from glmnet_py import glmnet, glmnetPredict; from glmnetPlot import glmnetPlot
from lifelines.utils import concordance_index
import json

def log_cpm(X):
    X = np.array([
        x/sum(x)
        for x in X
    ])
    X *= 10e6
    X = np.log(X+1)
    return X

def cross_fold_validation(df, folds, alpha, lambdas, generate_nulls=False):
    c_indices = []
    for f_i in range(len(folds)):
        test_indices = folds[f_i]
        train_indices = None
        for t_i in range(len(folds)):
            if t_i == f_i:
                continue
            if train_indices is None:
                train_indices = folds[t_i]
            else:
                train_indices = np.concatenate([
                    train_indices,
                    folds[t_i]
                ])

        print('Runing fold {}...'.format(f_i))
        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]

        X_train = np.array(
            df_train.drop(labels=['time', 'censor'], axis=1), 
            dtype=np.float64
        )
        Y_train = np.array(
            df_train[['time', 'censor']], 
            dtype=np.float64
        )
        X_test = np.array(
            df_test.drop(labels=['time', 'censor'], axis=1),
            dtype=np.float64
        )
        Y_test = np.array(
            df_test[['time', 'censor']], 
            dtype=np.float64
        )
        test_scores = _fit_and_predict(
            X_train = X_train,
            Y_train = Y_train,
            X_test = X_test,
            alpha=alpha,
            lambdas=lambdas
        )
        res = np.array([Y_test.T[0], np.squeeze(test_scores.T), Y_test.T[1]]).T
        sorted_surv_w_score = sorted(
            [
                (int(surv), score, cens)
                for surv, score, cens in res
            ],
            key=lambda x: x[1]
        )
        print(sorted_surv_w_score)

        #c_index = conc_index(test_scores, Y_test)
        c_index = concordance_index(
            Y_test.T[0],
            test_scores,
            event_observed=Y_test.T[1]
        )
        print('C-Index: ', c_index)
        c_indices.append(c_index)
    return c_indices


def _fit_and_predict(
        X_train,
        Y_train,
        X_test,
        alpha=0.0,
        lambdas=None
    ):
    fit = glmnet(
        x=X_train.copy(),
        y=Y_train.copy(),
        family='cox',
        alpha=alpha,
        standardize=True,
        lambdau=np.array(lambdas)
    )
    betas = fit['beta'].T
    pred_scores = -np.exp(np.dot(X_test, betas.T))
    return pred_scores

def empirical_pvalue(values, null_samples):
    sorted_nulls = sorted(null_samples, reverse=True)
    pvals = []
    for val in values:
        for null_i, null in enumerate(sorted_nulls):
            if null < val:
                pval = (null_i) / len(sorted_nulls)
                break
        pvals.append(pval)
    return pvals
