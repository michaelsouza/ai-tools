Variance-Aware Step Sizes for Stochastic Gradient Langevin
Dynamics on Nonconvex Objectives
Helena M. Duarte, Rafael T. Okafor, and Sun-Hee Park
Department of Applied Mathematics, Northgate Institute of Technology

1

Abstract
Stochastic gradient Langevin dynamics (SGLD) samples from a posterior by adding Gaussian
noise of variance 2ηk to each stochastic gradient step, but its accuracy depends on a step size ηk
that is usually tuned by hand. We propose a schedule that couples ηk to a running estimate σ̂k2 of
the gradient noise, so that the injected noise dominates the minibatch noise at every iteration. For
targets π(θ) ∝ e−U (θ) with L-smooth potentials satisfying a dissipativity condition, we bound the
√
1/2
2-Wasserstein distance between the law of the iterates and π by O( ηk d1/2 + σ̂k ηk ). Experiments
on Bayesian logistic regression and a two-layer network show that the schedule matches a tuned
constant step while removing the search over η.

I.

INTRODUCTION

Bayesian inference for large models requires samples from a posterior whose density is
known only up to a normalizing constant. Writing the target as π(θ) ∝ exp(−U (θ)) with θ ∈
√
Rd , the overdamped Langevin diffusion dθt = −∇U (θt ) dt + 2 dBt has π as its stationary
law, and its Euler–Maruyama discretization gives the unadjusted Langevin algorithm. When
U is a sum over N data points, each gradient costs O(N ) operations, which is prohibitive
for the data sets common in machine learning.
SGLD replaces the full gradient by an unbiased minibatch estimate and drops the
Metropolis correction.1 The update reads

d (θk ) +
θk+1 = θk − ηk ∇U
d (θk ) = N
where ∇U
n

P

p

2ηk ξk ,

ξk ∼ N (0, Id )

(1)

i∈Sk ∇Ui (θk ) is computed on a minibatch Sk of size n. Two sources

of error compete in this scheme. The discretization error grows with ηk , while the minibatch
noise has covariance of order ηk2 and becomes negligible relative to the injected noise only
when ηk is small. Practitioners therefore decrease ηk slowly or tune a constant step, and both
choices are made without information about the actual noise level of the gradient estimator.

1

A Metropolis step would require evaluating U (θ) on the full data set, which defeats the purpose of
subsampling.

2

·10−2
decreasing η0 /(1 + k/τ )
variance-aware

step size ηk

2
1.5
1
0.5
0
0

1,000

2,000

3,000

4,000

5,000

iteration k
FIG. 1. Step sizes produced by the variance-aware schedule and by a decreasing schedule with
η0 = 0.02 and τ = 400 during the first 5000 iterations of Bayesian logistic regression.
II.

VARIANCE-AWARE SCHEDULE

Our schedule estimates the trace of the gradient covariance with an exponential moving
average and shrinks the step whenever the minibatch noise would exceed a fixed fraction
d (θk ) and define the recursions
ρ ∈ (0, 1) of the injected noise. Let gk = ∇U

mk = β mk−1 + (1 − β) gk ,

(2)

vk = β vk−1 + (1 − β) ∥gk − mk−1 ∥2 ,
vk
σ̂k2 =
1 − βk

(3)
(4)

with m0 = 0, v0 = 0 and β ∈ [0.9, 0.999). The minibatch noise injected per step has
variance ηk2 σ̂k2 , whereas the Gaussian term contributes 2ηk d. Requiring the ratio of the two
to stay below ρ gives the step

ηk = min ηmax ,


2ρ d
(5)
σ̂k2 + ϵ
where ϵ > 0 prevents division by zero during the first iterations.2 The cap ηmax keeps
the discretization error under control in regions where the gradient noise is small but the
curvature is large.
2

In all experiments we use ϵ = 10−8 and ρ = 0.1; the results change by less than one standard error for
ρ ∈ [0.05, 0.2].

3

III.

CONVERGENCE ANALYSIS

We assume that U is L-smooth, that is, ∥∇U (x) − ∇U (y)∥ ≤ L∥x − y∥ for all x, y,
and that it is (m, b)-dissipative, meaning ⟨∇U (x), x⟩ ≥ m∥x∥2 − b. These conditions allow
nonconvex potentials and are standard in the analysis of Langevin algorithms. Under them,
the following bound holds for the law µk of the iterate θk .

P

W2 (µk , π) ≤ C1 e−λ j<k ηj W2 (µ0 , π)
p
√
+ C2 ηmax d + C3 max σ̂j ηj

(6)

j<k

The constants C1 , C2 and C3 depend on L, m and b but not on the dimension, and λ is
the contraction rate of the continuous diffusion in the Wasserstein metric. The first term
vanishes as the accumulated step length grows, so the schedule converges provided that
P
2
k ηk = ∞, which holds because ηk is bounded below whenever σ̂k stays bounded.
The proof splits the error into a discretization part and a stochastic part. For the latter
we write the minibatch error as a martingale difference ζk = gk − ∇U (θk ) and use the
piecewise definition of the step to separate the two regimes of the schedule:

E∥ηk ζk ∥2 ≤



η 2 σ̂ 2 , σ̂ 2 ≤ 2ρd/ηmax ,
max k
k

2ρd ηk ,

(7)

otherwise.

In both regimes the stochastic error is at most 2ρdηk , which is a fraction ρ of the injected
noise. Summing over iterations and applying the contraction of the diffusion yields the
bound.

IV.

EXPERIMENTS

We compare the schedule with a tuned constant step and with the decreasing schedule
ηk = η0 (1 + k/τ )−1 on two problems. The first is Bayesian logistic regression on a data set
with N = 58,000 examples and d = 54 features. The second is a two-layer network with 256
hidden units, for which the posterior is multimodal. Accuracy is measured by the kernel
Stein discrepancy of 2000 retained samples and by the predictive log-likelihood on held-out
data.
4

On logistic regression the variance-aware schedule reaches a Stein discrepancy of 0.031 ±
0.002, against 0.029 ± 0.002 for the best constant step found by a grid of eight values and
0.047 ± 0.004 for the decreasing schedule. On the network, the predictive log-likelihood
is −0.412 for our schedule and −0.409 for the tuned constant step, while the decreasing
schedule stalls at −0.455 because its steps become too small before the chain leaves the
initial mode.

V.

CONCLUSION

Coupling the step size of SGLD to an online estimate of the gradient noise removes the
most expensive tuning parameter of the method at no cost in accuracy on the problems we
tested. The analysis extends to preconditioned variants in which σ̂k2 is replaced by a diagonal
estimate, a direction we leave for future work.

5

