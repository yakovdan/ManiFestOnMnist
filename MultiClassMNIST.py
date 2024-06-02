from oct2py import octave
import cupy as cp
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

from tools import *

STORE_MODE = True
DISABLE_SAVE = True
VISUALIZE = True
octave.addpath('./SPSD')


# General Params
random_state = 40
np.random.seed(random_state)

# load MNIST dataset 4 and 9 digits
(X, y), (_, _) = mnist.load_data()

# SVM hyperparameters search space
C_range = np.arange(-5, 16, 3)
C_range = np.power(2 * np.ones_like(C_range, dtype=np.float64), C_range)

gamma_range = np.arange(-15, 6, 3)
gamma_range = np.power(2 * np.ones_like(gamma_range, dtype=np.float64), gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)

# list for tracking performance on train and test

test_accuracies = []
train_accuracies = []
for exp_idx in range(1):  # Folds for picking optimal percentile. Can be 1.

    # Randomize data
    idx = np.arange(60000)
    np.random.shuffle(idx)
    x_shuff, y_shuff = X[idx], y[idx]
    X, y = x_shuff, y_shuff

    # Choose 300 "random" examples from our dataset
    x_train, x_test = np.zeros(shape=(3000, 28, 28), dtype=X.dtype), np.zeros(shape=(57000, 28, 28), dtype=X.dtype)
    y_train, y_test = np.zeros(shape=(3000,), dtype=y.dtype), np.zeros(shape=(57000,), dtype=y.dtype)

    # reshape dataset
    position = 0
    for i in range(10):
        elem_count = (y == i).sum()
        test_count = elem_count - 300
        x_train[300*i: 300*(i+1)] = X[y == i][:300]
        y_train[300*i: 300*(i+1)] = i
        x_test[position: position+test_count] = X[y == i][300:]
        y_test[position: position+test_count] = i
        position += test_count


    # folds for SVM
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    best_accuracy = 0
    best_features_idx = None
    best_percentile_for_estimator = None
    best_estimator = None
    best_params = None
    PERCENTILES: list[int] = [30, 50, 70, 90, 95, 99, 100]  # Used to pick a scale factor for RBF kernel
    for cur_percentile in PERCENTILES:
        print(f"Percentile {cur_percentile}")
        # Compute kernels
        if STORE_MODE:
            kernels = construct_kernel(x_train, y_train, cur_percentile)

            # Compute minimal rank in our kernels
            min_rank = 20 # min([np.linalg.matrix_rank(K, tol=5e-3) for K in kernels])
            print(f"min_rank: {min_rank}")
            # Compute M from kernels
            kernels_oct = np.stack(kernels, axis=2)
            kernels_cp = cp.array(np.stack(kernels))
            N = kernels_cp.shape[0]

            # calculate the geometric mean of SPSD matrices by calling into octave code
            # M - SOSD mean; mG - Grassmannian mean; mP - mean of SPD component;
            # UU - eigenbasis for each difference operator
            M, mG, mP, UU, TT = octave_spsd_mean(kernels_oct, min_rank)

            # calculate the difference operator for each class. Each operator by definition is a SPSD matrix
            # D_ - difference operator
            G_, _, _, _, _ = spsd_geodesics(M, kernels_cp, 1, min_rank)
            logP_ = log_map(mP[None, :], TT)
            D_ = symmetrize(G_ @ logP_ @ cp.swapaxes(G_, -1, -2))

            # calculate eigen decomposition for each difference operator

            eigvals, eigvecs = cp.linalg.eigh(D_)
            eigvecs_square = cp.square(eigvecs)
            eigvals_abs = cp.expand_dims(cp.abs(eigvals), axis=1)

            # calculate ManiFest score

            r = eigvals_abs * eigvecs_square
            r = r.sum(axis=2)
            score = cp.max(r, axis=0)
            idx = cp.argsort(score, axis=0)[::-1]

            if not DISABLE_SAVE:
                save_matrices(cur_percentile, M, mG, mP, UU, TT, G_, D_, score, idx, kernels_cp)
        else:
            M, mG, mP, UU, TT, G_, D_, score, idx = load_matrices(cur_percentile)
            eigvals, eigvecs = cp.linalg.eigh(D_)
            eigvecs_square = cp.square(eigvecs)
            eigvals_abs = cp.expand_dims(cp.abs(eigvals), axis=1)

        # pick 50 features with largest eigenvalues in absolute value

        idx_top50 = idx[:50]
        top50_features = [(x // 28, x % 28) for x in idx_top50]
        score_viz = cp.abs(score).get().reshape((1, 28, 28))
        score_viz_sq = np.square(score_viz)
        if VISUALIZE:
            visualize_digit(score_viz, 0, top50_features, some_title=f"score_{cur_percentile}", mode =0)
            visualize_digit(score_viz_sq, 0, top50_features, some_title=f"score_sq_{cur_percentile}", mode=0)
            for class_idx in range(10):
                eig_vec_for_viz = cp.abs(eigvecs[class_idx, :, -1]).get().reshape((1, 28, 28))
                visualize_digit(eig_vec_for_viz, 0, top50_features, some_title=f"{class_idx}_{cur_percentile}")
                eig_vec_for_viz = np.square(eig_vec_for_viz)
                visualize_digit(eig_vec_for_viz, 0, top50_features, some_title=f"{class_idx}_sq_{cur_percentile}")

        # generate a new train dataset by picking selected features only
        x_fs = x_train.reshape(x_train.shape[0], -1)[:, idx_top50.get()]
        x_fs = x_fs / x_fs.max()

        # train and validate SVM with 10 folds for cross validation
        grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv, scoring="accuracy", verbose=2, n_jobs=4)
        y_fs = y_train.astype(np.int8)
        grid.fit(x_fs, y_fs)
        if grid.best_score_ > best_accuracy:
            best_accuracy = grid.best_score_
            best_percentile_for_estimator = cur_percentile
            best_estimator = grid.best_estimator_
            best_features_idx = idx_top50
            best_params = grid.best_params_
            cp.save('bestM', M)
            cp.save('bestD', D_)
            print("Found new best SVM")
        print(
            f"Percentile: {cur_percentile}\t The best parameters are {grid.best_params_} with a score of %{ 100* grid.best_score_:.5f}")
        train_accuracies.append(grid.best_score_)

    # evaluate on test set
    x_test_fs = x_test.reshape(x_test.shape[0], -1)[:, best_features_idx.get()]
    y_test_fs = y_test.astype(np.int8)
    x_test_fs = x_test_fs / x_test_fs.max()
    y_target = best_estimator.predict(x_test_fs)
    accuracy = (y_test_fs == y_target).sum() / y_target.shape[0]
    print(f"Final accuracy: {accuracy}")
    test_accuracies.append(accuracy)
    print(f"train acc: {train_accuracies}, mean: {np.mean(np.array(train_accuracies))}, std: {np.std(np.array(train_accuracies))}")

print(f"test acc: {test_accuracies}, mean: {np.mean(np.array(test_accuracies))}, std: {np.std(np.array(test_accuracies))}")