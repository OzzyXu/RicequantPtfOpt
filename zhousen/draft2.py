import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)




def is_pd(K):
    try:
        np.linalg.cholesky(K)
        return 1
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0
        else:
            raise


A = np.matrix([[1,0.9,1], [0.9,1,0.9],[1,0.9,1]])
A

A = np.matrix([[1,0.1,0.1],[0.1,1,1],[0.1,1,1]])

A
is_pos_def(A)

np.linalg.eigvals(A)

is_pd(A)