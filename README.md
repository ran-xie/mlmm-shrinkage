# Nonparametric Shrinkage Estimation of High-Dimensional Covariance Matrices within Random Effects Models

## Installation

```bash
git clone https://github.com/ran-xie/mlmm-shrinkage.git
```

## Model Setup

We study multi-trait linear mixed effects models in a high-dimensional regime, where the number of traits $p$ may be comparable to or exceed the number of individuals $n$.

### Model MLMM

(i) Consider a $k$-level multi-trait linear mixed model of the form

$$ Y = X\beta + \sum_{r=1}^k U_r \alpha_r, \quad \alpha_r \sim N(0, \text{Id}_{I_r} \otimes \Sigma_r),$$

where $Y \in \mathbb{R}^{n \times p}$ is the observed trait matrix, $X \in \mathbb{R}^{n \times I_0}$ is the fixed effects design matrix, $\beta \in \mathbb{R}^{I_0 \times p}$ is the fixed effect coefficient matrix, $U_r \in \mathbb{R}^{n \times I_r}$ are known design matrices, and $\alpha_r \in \mathbb{R}^{I_r \times p}$ are independent random effect matrices.

(ii) Let  $B \in \mathbb{R}^{n \times n}$ be a non-zero matrix that satisfies $BX = 0$. Our estimators are built from quadratic form matrices

$$
S_n = p^{-1} Y^T B Y.
$$

### Estimating $\Sigma_1,\ldots,\Sigma_k$

Without loss of generality, we focus on $\Sigma_1$. Given a quadratic-form estimator $S_n$ (e.g., an unbiased MANOVA estimator), we study shrinkage estimators with squared Frobenius loss

$$
    \mathcal{L}*n^{\text{FR}}(\hat{\Sigma}n,\Sigma_1)=\frac{1}{p}\Tr [(\hat{\Sigma}n-\Sigma_1)^2].
$$

We consider estimators that share the eigenvectors of $S_n$ and shrink the eigenvalues. The oracle estimator in this class has $i$-th eigenvalue

$$
d{n,i}^{\text{or}}=u{n,i}^T\Sigma_1u*{n,i},
$$

where $u_{n,i}$ is the $i$-th eigenvector of $S_n$.

This package implements:


| Symbol                     | Description                                                           |
| -------------------------- | --------------------------------------------------------------------- |
| $\check{d}_n$              | Deterministic-equivalent oracle (requires $\Sigma_1,\ldots,\Sigma_k$) |
| $\hat{d}_n$                | Bona fide nonparametric estimator                                     |
| $\hat{d}_n^{\mathrm{iso}}$ | Isotonic regression applied to $\hat{d}_n$                            |


## Usage

Inputs: $n$ by $p$ obeservation matrix `Y`, a list of $k$ incidence matrices `U_list=[U_1,...,U_k]`, and a determinsitic matrix `B` with $BX = 0$ fixed a priori; for the oracle, the list of $k$ population covariance matrices `Sigma_list=[Sigma_1,...,Sigma_k]`.

```python
from Nonparametric.nonparametric import deteqv_oracle_est, bona_fide_est

d_check,_,_ = deteqv_oracle_est(Y, Sigma_list, U_list, B)
d_hat, d_iso,_,_ = bona_fide_est(Y, U_list, B)
```

For a simplified version of the algorithm specialized to full sibling models, see 'experiments/fullsib.ipynb'.

## Reproducible Research

The code available under 'experiments/' in the repository replicates the experimental results in our paper.

## References

```bibtex
@article{Nonparametric2026, 
  author={Iain Johnstone and Ran Xie},
  title={Nonparametric shrinkage estimation of high-dimensional genetic covariance},  
  year={2026}, 
  ...
}
```

