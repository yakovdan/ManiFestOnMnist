import cupy as cp
import numpy as np
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from oct2py import octave

def symmetrize(mat):
    return (cp.swapaxes(mat, -1, -2) + mat) / 2


def random_spd_matrix(n: int, cond_num_limit: int = 1000):
    mat = cp.random.rand(n,n)
    eig_vals, _ = cp.linalg.eig(mat)
    r = (eig_vals != 0).sum()
    if r == n:
        mat = mat.T @ mat
        eig_vals, eig_vecs = cp.linalg.eigh(mat)
        largest_eig = cp.max(eig_vals)
        for i in range(len(eig_vals)):
            if eig_vals[i] < largest_eig / cond_num_limit:
                eig_vals[i] += largest_eig / cond_num_limit
        return eig_vecs @ cp.diag(eig_vals) @ eig_vecs.T
    raise RuntimeError("Failed creating a random SPD matrix")


def matrix_pow(mat, p: float):
    eig_vals, eig_vecs = cp.linalg.eigh(mat)
    diag = cp.zeros_like(eig_vecs)
    idx = cp.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = cp.power(eig_vals, p)
    return eig_vecs @ diag @ cp.swapaxes(eig_vecs, -1, -2)


def matrix_exp(mat):
    eig_vals, eig_vecs = cp.linalg.eigh(mat)
    diag = cp.zeros_like(eig_vecs)
    idx = cp.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = cp.exp(eig_vals)
    return eig_vecs @ diag @ cp.swapaxes(eig_vecs, -1, -2)


def matrix_log(mat):
    eig_vals, eig_vecs = cp.linalg.eigh(mat)
    diag = cp.zeros_like(eig_vecs)
    idx = cp.arange(eig_vals.shape[-1])
    diag[:, idx, idx] = cp.log(eig_vals)
    return eig_vecs @ diag @ cp.swapaxes(eig_vecs, -1, -2)


def exponential_map(p1, p2):
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    expmap_mat = p1_power_half @ matrix_exp(p1_power_minus_half @ p2 @ p1_power_minus_half) @ p1_power_half
    return expmap_mat


def log_map(p1, p2):
    p1_power_half = matrix_pow(symmetrize(p1), 0.5)
    p1_power_minus_half = matrix_pow(symmetrize(p1), -0.5)
    logmap_mat = p1_power_half @ matrix_log(symmetrize(p1_power_minus_half @ p2 @ p1_power_minus_half)) @ p1_power_half
    return logmap_mat


def spd_geodesic(p1, p2, t: float):
    p1_power_half = matrix_pow(p1, 0.5)
    p1_power_minus_half = matrix_pow(p1, -0.5)
    return p1_power_half @ matrix_pow(p1_power_minus_half @ p2 @ p1_power_minus_half, t) @ p1_power_half


def spd_matrix_mean(matrices, iter_limit: int = 200, eps: float = 1e-12):
    print(f"Input matrix norms: {cp.linalg.norm(matrices, ord='fro', axis=(1, 2))}")
    N = matrices.shape[0]
    spd_mean = symmetrize(cp.sum(matrices, axis=0, keepdims=True) / N)

    norm_val = 1
    count = 0
    while norm_val > eps and count < iter_limit:
        sum_projections = log_map(spd_mean, matrices)
        mean_projections = symmetrize(cp.sum(sum_projections, axis=0) / N)
        spd_mean = symmetrize(exponential_map(spd_mean, mean_projections))
        norm_val = cp.linalg.norm(mean_projections, ord='fro')
        print(count, norm_val)
        count += 1
    print(count)
    return spd_mean


def calc_tol(matrix, var_type='float64', energy_tol=0):
    tol = np.max(matrix) * len(matrix) * np.core.finfo(var_type).eps
    tol2 = np.sqrt(np.sum(matrix ** 2) * energy_tol)
    tol = np.max([tol, tol2])

    return tol

def construct_multidiag(arr):
    assert arr.ndim == 2
    diag_out = cp.zeros(shape=(arr.shape[0], arr.shape[1], arr.shape[1]))
    id = cp.arange(arr.shape[1])
    diag_out[:, id, id] = arr[:, id]
    return diag_out

def spsd_geodesics(G1, G2, p=0.5, r=None, eigVecG1=None, eigValG1=None, eigVecG2=None, eigValG2=None):
    if eigVecG1 is None:
        eigValG1, eigVecG1 = cp.linalg.eigh(G1)
    if eigVecG2 is None:
        eigValG2, eigVecG2 = cp.linalg.eigh(G2)

    if r is None:
        tol = calc_tol(eigValG1)
        rank_G1 = len(cp.abs(eigValG1)[cp.abs(eigValG1) > tol])

        tol = calc_tol(eigValG2)
        rank_G2 = len(cp.abs(eigValG2)[cp.abs(eigValG2) > tol])

        r = min(rank_G1, rank_G2)

    maxIndciesG1 = cp.flip(cp.argsort(cp.abs(eigValG1))[-r:], 0)
    V1 = eigVecG1[:, maxIndciesG1]
    lambda1 = eigValG1[maxIndciesG1]

    maxIndciesG2 = cp.flip(cp.argsort(cp.abs(eigValG2))[:, -r:], 1)
    lambda2 = cp.take_along_axis(eigValG2, maxIndciesG2, 1)
    maxIndciesG2 = cp.expand_dims(maxIndciesG2, 1)
    V2 = cp.take_along_axis(eigVecG2, maxIndciesG2, axis=2)

    O2, sigma, O1T = cp.linalg.svd(cp.swapaxes(V2, -1, -2) @ V1)
    O1 = cp.swapaxes(O1T, -1, -2)

    sigma[sigma < -1] = -1
    sigma[sigma > 1] = 1
    theta = cp.arccos(sigma)

    U1 = V1 @ O1
    R1 = cp.swapaxes(O1, -1, -2) @ cp.diag(lambda1) @ O1

    lambda2_diag = construct_multidiag(lambda2)
    U2 = V2 @ O2
    R2 = cp.swapaxes(O2, -1, -2) @ lambda2_diag @ O2

    tol = calc_tol(sigma.get())
    valid_ind = cp.where(cp.abs(sigma - 1) > tol)
    pinv_sin_theta = cp.zeros(theta.shape)
    pinv_sin_theta[valid_ind] = 1 / cp.sin(theta[valid_ind])

    UG1G2 = U1 @ construct_multidiag(cp.cos(theta * p)) + (cp.eye(G1.shape[0]) - U1 @ cp.swapaxes(U1, -1, -2)) @ U2 @ construct_multidiag(
        pinv_sin_theta) @ construct_multidiag(cp.sin(theta * p))

    return UG1G2, R1, R2, O1, lambda1


def construct_kernel(X, y, percentile=50):
    labels = list(set(y))
    kernels = []

    for i in range(len(labels)):
        elements = X[y == i].shape[0]
        x = X[y == i].reshape(elements, -1)
        K_dis = euclidean_distances(np.transpose(x))
        #epsilon = np.percentile(K_dis[~np.eye(K_dis.shape[0], dtype=bool)], percentile)
        perc = np.percentile(K_dis[~np.eye(K_dis.shape[0], dtype=bool)], percentile)
        #med = np.median(K_dis[~np.eye(K_dis.shape[0], dtype=bool)])
        epsilon = perc

        K = np.exp(-(K_dis ** 2) / (2 * epsilon ** 2))
        kernels.append(K)
    return kernels


def visualize_digit(digit_array: np.ndarray, digit_idx: int, feature_coords, resize_factor: int = 1, some_title="",
                    mode=1) -> None:
    # mode = 0 to draw a score
    # mode = 1 to draw an eigenvector
    # differences are the colors of the circles that highlight selected features and the use of transparency

    fig, ax = plt.subplots(1)

    # Show the image
    digit = digit_array[digit_idx, :].reshape(28, 28)
    ax.set_title(some_title)
    ax.axis('off')
    ax.imshow(digit)

    if mode == 0:
        # Now, loop through coord arrays, and create a circle at each x,y pair
        for yy, xx in feature_coords[:20]:
            circ = Circle((resize_factor * xx, resize_factor * yy), 0.5, color='orange', fill=False, linewidth=2.0)
            ax.add_patch(circ)

        for yy, xx in feature_coords[20:]:
            circ = Circle((resize_factor * xx, resize_factor * yy), 0.5, color='red', fill=False, linewidth=2.0)
            ax.add_patch(circ)
    else:

        for yy, xx in feature_coords:
            circ = Circle((resize_factor * xx, resize_factor * yy), 0.5, color='red', fill=False, alpha=0.5)
            ax.add_patch(circ)
    filename = some_title.replace(" ", "_") + ".png"
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)

    # As a default, figures are saved to disk
    #plt.show()
    plt.close(fig)


def octave_spsd_mean(kernels_for_octave, min_rank):
    #### calculate SPSD mean using Octave
    octave.eval("""function [mC, mG, mP, UU, TT] = roundtrip(y, req_rank)
    %
    l = size(y, 3);
    CC{l} = [];
    for i=1:l
        CC{i} = y(:, :, i);
    end
    size(CC);
    size(CC{1});
    [mC, mG, mP, UU, TT] = SpsdMean(CC, req_rank);
    """)

    M, mG, mP, UU, TT = octave.roundtrip(kernels_for_octave, min_rank, nout=5)
    M = cp.array(M)
    mG = cp.array(mG)
    mP = cp.array(mP)
    UUU = np.concatenate([UU.item(i)[None, :] for i in range(UU.size)], axis=0)
    TTT = np.concatenate([TT.item(i)[None, :] for i in range(TT.size)], axis=0)
    UU = cp.array(np.copy(UUU))
    TT = cp.array(np.copy(TTT))

    return M, mG, mP, UU, TT


def save_matrices(cur_percentile, M, mG, mP, UU, TT, G_, D_, score, idx, kernels_cp):
    cp.save(f'M_{cur_percentile}_1', M)
    cp.save(f'mG_{cur_percentile}_1', mG)
    cp.save(f'mP_{cur_percentile}_1', mP)
    cp.save(f'UU_{cur_percentile}_1', UU)
    cp.save(f'TT_{cur_percentile}_1', TT)
    cp.save(f'G_{cur_percentile}_1', G_)
    cp.save(f'D_{cur_percentile}_1', D_)
    cp.save(f'score_{cur_percentile}_1', score)
    cp.save(f'idx_{cur_percentile}_1', idx)
    cp.save(f'kernels_{cur_percentile}_1', kernels_cp)

def load_matrices(cur_percentile):
    M = cp.load(f'M_{cur_percentile}_1.npy')
    mG = cp.load(f'mG_{cur_percentile}_1.npy')
    mP = cp.load(f'mP_{cur_percentile}_1.npy')
    UU = cp.load(f'UU_{cur_percentile}_1.npy')
    TT = cp.load(f'TT_{cur_percentile}_1.npy')
    G_ = cp.load(f'G_{cur_percentile}_1.npy')
    D_ = cp.load(f'D_{cur_percentile}_1.npy')
    score = cp.load(f'score_{cur_percentile}_1.npy')
    idx = cp.load(f'idx_{cur_percentile}_1.npy')
    kernels_cp = cp.load(f'kernels_{cur_percentile}_1.npy')

    return M, mG, mP, UU, TT, G_, D_, score, idx, kernels_cp