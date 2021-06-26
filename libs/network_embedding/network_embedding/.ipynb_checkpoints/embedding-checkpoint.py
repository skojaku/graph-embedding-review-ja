import numpy as np
from scipy import sparse


def embed_network(A, dim, window_length=5):
    """
    Embed network using the DeepWalk baesd on a matrix decomposition:

    Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., & Tang, J. (2018).
    Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and Node2vec.
    Proceedings of the Eleventh ACM International Conference on
    Web Search and Data Mining, 2018-Febua, 459–467.
    
    Paraeters
    ---------
    A: scipy.sparse matrix
        Adjacency matrix. Should be symmetric (accept only undirected networks)
    dim: int
        Dimension of embedding space
    window_length: int
        number of steps that a random walker walks.

    Return
    ------
    vec: numpy.ndarray
        In-vector
    """
    N = A.shape[0]  # number of nodes

    # Construct the transition matrix
    deg = np.array(np.sum(A, axis=1)).reshape(-1)
    P = sparse.diags(1 / np.maximum(deg, 1) ) @ A
    # Run the random walk window_length steps
    Pt = P.copy()
    for i in range(window_length - 1):
        Pt = P @ (Pt + sparse.diags(np.ones(N)))
    Pt = Pt / window_length

    # Re-normalize the row
    W = sparse.diags(deg) @ Pt

    # Set a small value (wmin) to the zero transition probability to
    # prevent under floating
    wmin = 1e-8
    w = np.array(np.sum(W, axis=1)).reshape(-1)
    logw = safe_log(w)
    logW = W.copy()
    logW.data = safe_log(logW.data) - safe_log(wmin)

    # Singular value decomposition
    mat_seq = [
        [logW],
        [safe_log(wmin) * np.ones((N, 1)), np.ones((1, N))],
        # [np.log(np.sum(W)) * np.ones((N, 1)), np.ones((1, N))],
        [-logw.reshape((N, 1)), np.ones((1, N))],
        [-np.ones((N, 1)), logw.reshape((1, N))],
    ]

    c = mat_prod_matrix_seq(mat_seq, np.ones((N, 1)))
    # mat_seq += [[c/N, np.ones((1, N))]]
    mat_seq += [[-np.sum(c) / (N * N) * np.ones((N, 1)), np.ones((1, N))]]
    vec, s, _ = rSVD_for_decomposable_matrix(mat_seq, dim)

    # Set the scale of embedding space
    vec = vec @ np.diag(np.sqrt(s))
    return vec


#
# Helper function
#
def rSVD_for_decomposable_matrix(mat_seq, r, p=10, q=1, fill_zero=1e-20):
    """
    Randomized SVD for decomposable matrix.
    We assume that the matrix is given by mat_seq[0] + mat_seq[1],...

    Parameters
    ----------
    mat_seq: list
        List of decomposed matrices
    r : int
        Rank of decomposed matrix
    p : int (Optional; Default p = 10)
        Oversampling
    q : int (Optional; Default q = 1)
        Number of power iterations
    fill_zero: float
        Replace the zero values in the transition matrix with this value.

    Return
    ------
    U : numpy.ndrray
        Left singular vectors of size (X.shape[0], r)
    lams : numpy.ndarray
        Singular values of size (r,)
    V : numpy.ndarray
        Right singular vectors of size (X.shape[0], r)
    """
    Nc = mat_seq[-1][-1].shape[1]
    dim = r + p

    R = np.random.randn(Nc, dim)  # Random gaussian matrix
    Z = mat_prod_matrix_seq(mat_seq, R)
    # Z = X @ R

    for i in range(q):  # Power iterations
        zz = mat_prod_matrix_seq(Z.T, mat_seq)
        Z = mat_prod_matrix_seq(mat_seq, zz.T)
    Q, R = np.linalg.qr(Z, mode="reduced")

    Y = mat_prod_matrix_seq(Q.T, mat_seq)

    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    selected = np.argsort(np.abs(S))[::-1][0:r]
    return U[:, selected], S[selected], VT[selected, :]


def mat_prod_matrix_seq(A, B):
    def right_mat_prod_matrix_seq(A, matrix_seq):

        S = None
        for k in range(len(matrix_seq)):
            R = A @ matrix_seq[k][0]
            if sparse.issparse(R):
                R = R.toarray()

            for l in range(1, len(matrix_seq[k])):
                R = R @ matrix_seq[k][l]

            if S is None:
                S = R
            else:
                S = S + R
        return S

    def left_mat_prod_matrix_seq(matrix_seq, A):

        S = None
        for k in range(len(matrix_seq)):
            R = matrix_seq[k][-1] @ A
            if sparse.issparse(R):
                R = R.toarray()

            for l in range(1, len(matrix_seq[k])):
                R = matrix_seq[k][-l - 1] @ R

            if S is None:
                S = R
            else:
                S = S + R
        return S

    if isinstance(A, list) and not isinstance(B, list):
        return left_mat_prod_matrix_seq(A, B)
    elif isinstance(B, list) and not isinstance(A, list):
        return right_mat_prod_matrix_seq(A, B)

def safe_log(x, minval = 1e-20):
    return np.log(np.maximum(x, minval))
