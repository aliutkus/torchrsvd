import torch
from torchrsvd import rsvd

if __name__ == "__main__":
    # test randomized SVD on a small low-rank matrix
    import torch
    rank=10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.randn(2000, rank).to('cuda')
    b = torch.randn(rank, 2000).to('cuda')
    input = torch.matmul(a, b)

    (m, n) = input.shape
    print('SVD on (%d, %d) input' % (m,n)) 
    import time

    start = time.time()

    (ufull, sfull, vfull) = torch.svd(input)
    print('   torch.svd: %0.1fms' % (1000*(time.time()-start)) )
    start=time.time()
    (u, s, v)=rsvd(input, rank=rank)
    print('   rsvd (%d components): %0.1fms' % (rank, 1000*(time.time()-start)))
    print('errors:')
    reconstructed_full = torch.matmul(
        ufull[:,:rank], torch.matmul(torch.diag(sfull[:rank]), vfull[:,:rank].T))
    reconstructed_rsvd = torch.matmul(u, torch.matmul(torch.diag(s), v.T)) 
    print('   fast vs truncated full: %f'%torch.norm(
        reconstructed_full-reconstructed_rsvd))
    print('   input vs fast: %f' % torch.norm(
        input-reconstructed_rsvd))
