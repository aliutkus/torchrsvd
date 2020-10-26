import torch
import contextlib


def rsvd(input, rank):
    """
    Randomized SVD torch function

    Extremely fast computation of the truncated Singular Value Decomposition, using
    randomized algorithms as described in Halko et al. 'finding structure with randomness

    usage :

    Parameters:
    -----------
    * input : Tensor (2D matrix) whose SVD we want
    * rank : (int) number of components to keep

    Returns:
    * (u,s,v) : tuple, classical output as the builtin torch svd function
    """
    assert len(input.shape) == 2, "input tensor must be 2D"
    (m, n) = input.shape
    p = torch.min(torch.tensor([2 * rank, n]))
    x = torch.randn(n, p, device=input.device)
    y = torch.matmul(input, x)

    # get an orthonormal basis for y
    uy, sy, _ = torch.svd(y)
    rcond = torch.finfo(input.dtype).eps * m
    tol = sy.max() * rcond
    num = torch.sum(sy > tol)
    W1 = uy[:, :num]

    B = torch.matmul(W1.T, input)
    W2, s, v = torch.svd(B)
    u = torch.matmul(W1, W2)
    k = torch.min(torch.tensor([rank, u.shape[1]]))
    return(u[:, :k], s[:k], v[:, :k])

if __name__ == "__main__":
    # test randomized SVD on a small low-rank matrix
    import torch
    rank=10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.randn(2000, rank).to('cuda')
    b = torch.randn(rank, 2000).to('cuda')
    input = torch.matmul(a, b)

    import time
    start = time.time()

    (ufull, sfull, vfull) = torch.svd(input)
    print('torch.svd: %0.1fms' % (1000*(time.time()-start)) )
    start=time.time()
    (u, s, v)=rsvd(input, rank=rank)
    print('rsvd: %0.1fms' % (1000*(time.time()-start)))

    print('errors:')
    reconstructed_full = torch.matmul(
        ufull[:,:rank], torch.matmul(torch.diag(sfull[:rank]), vfull[:,:rank].T))
    reconstructed_rsvd = torch.matmul(u, torch.matmul(torch.diag(s), v.T)) 
    print('fast vs truncated full: %f'%torch.norm(
        reconstructed_full-reconstructed_rsvd))
    print('input vs fast: %f' % torch.norm(
        input-reconstructed_rsvd))
