# torchrsvd
## fast truncated randomized SVD for pytorch

## Presentation

This repository implements a `rsvd(input, rank)` function that computes truncated singular value decomposition (SVD) through random projections, as described in 
> _Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decomposition_, N. Halko et al. arXiv:0909.4061

## Installation

Type `pip install -e .` in the root folder of this repo.

## Usage

Try out `python test.py` in the `examples` folder.

```
SVD on (2000, 2000) input
   torch.svd: 1293.4ms
   rsvd (10 components): 19.3ms
errors:
   fast vs truncated full: 0.009194
   input vs fast: 0.005163
```

