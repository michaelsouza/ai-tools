Variance-Aware Step Sizes for Stochastic Gradient Langevin
Dynamics on Nonconvex Objectives
Helena M. Duarte

Northgate Institute of Technology
USA

Rafael T. Okafor

Abstract
Stochastic gradient Langevin dynamics (SGLD) samples from a posterior by adding Gaussian noise of variance 2𝜂𝑘 to each stochastic
gradient step, but its accuracy depends on a step size 𝜂𝑘 that is
usually tuned by hand. We propose a schedule that couples 𝜂𝑘 to
a running estimate 𝜎ˆ𝑘2 of the gradient noise, so that the injected
noise dominates the minibatch noise at every iteration. For targets
𝜋 (𝜃 ) ∝ 𝑒 −𝑈 (𝜃 ) with 𝐿-smooth potentials satisfying a dissipativity condition, we bound the 2-Wasserstein distance between the
√
law of the iterates and 𝜋 by 𝑂 ( 𝜂𝑘 𝑑 1/2 + 𝜎ˆ𝑘 𝜂𝑘1/2 ). Experiments on
Bayesian logistic regression and a two-layer network show that the
schedule matches a tuned constant step while removing the search
over 𝜂.

2

Variance-Aware Schedule

Our schedule estimates the trace of the gradient covariance with
an exponential moving average and shrinks the step whenever the
minibatch noise would exceed a fixed fraction 𝜌 ∈ (0, 1) of the
c (𝜃 𝑘 ) and define the recursions
injected noise. Let 𝑔𝑘 = ∇𝑈
(2)

𝑚𝑘 = 𝛽 𝑚𝑘 −1 + (1 − 𝛽) 𝑔𝑘 ,

ACM Reference Format:
Helena M. Duarte, Rafael T. Okafor, and Sun-Hee Park. 2026. VarianceAware Step Sizes for Stochastic Gradient Langevin Dynamics on Nonconvex
Objectives. In Symposium on Statistical Learning and Computation (SLC ’26).
ACM, New York, NY, USA, 2 pages. https://doi.org/10.5555/3741104.3741141

Introduction

Bayesian inference for large models requires samples from a posterior whose density is known only up to a normalizing constant.
Writing the target as 𝜋 (𝜃 ) ∝ exp(−𝑈 (𝜃 )) with 𝜃 √∈ R𝑑 , the overdamped Langevin diffusion d𝜃 𝑡 = −∇𝑈 (𝜃 𝑡 ) d𝑡 + 2 d𝐵𝑡 has 𝜋 as
its stationary law, and its Euler–Maruyama discretization gives
the unadjusted Langevin algorithm. When 𝑈 is a sum over 𝑁 data
points, each gradient costs 𝑂 (𝑁 ) operations, which is prohibitive
for the data sets common in machine learning.
SGLD replaces the full gradient by an unbiased minibatch estimate and drops the Metropolis correction.1 The update reads
√︁
c (𝜃 𝑘 ) + 2𝜂𝑘 𝜉𝑘 ,
𝜃 𝑘+1 = 𝜃 𝑘 − 𝜂𝑘 ∇𝑈

Northgate Institute of Technology
USA

c (𝜃 𝑘 ) = 𝑁 Í𝑖 ∈𝑆 ∇𝑈𝑖 (𝜃 𝑘 ) is computed on a minibatch
where ∇𝑈
𝑛
𝑘
𝑆𝑘 of size 𝑛. Two sources of error compete in this scheme. The
discretization error grows with 𝜂𝑘 , while the minibatch noise has
covariance of order 𝜂𝑘2 and becomes negligible relative to the injected noise only when 𝜂𝑘 is small. Practitioners therefore decrease
𝜂𝑘 slowly or tune a constant step, and both choices are made without
information about the actual noise level of the gradient estimator.

Keywords
Langevin dynamics, Bayesian inference, step-size adaptation, Wasserstein distance

1

Sun-Hee Park

Northgate Institute of Technology
USA

𝜉𝑘 ∼ N (0, 𝐼𝑑 )

(1)

1 A Metropolis step would require evaluating 𝑈 (𝜃 ) on the full data set, which defeats

the purpose of subsampling.

Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
SLC ’26, Fortaleza, Brazil
© 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/10.5555/3741104.3741141

2

𝑣𝑘 = 𝛽 𝑣𝑘 −1 + (1 − 𝛽) ∥𝑔𝑘 − 𝑚𝑘 −1 ∥ ,
(3)
𝑣𝑘
2
𝜎ˆ𝑘 =
(4)
1 − 𝛽𝑘
with 𝑚 0 = 0, 𝑣 0 = 0 and 𝛽 ∈ [0.9, 0.999). The minibatch noise
injected per step has variance 𝜂𝑘2 𝜎ˆ𝑘2 , whereas the Gaussian term
contributes 2𝜂𝑘 𝑑. Requiring the ratio of the two to stay below 𝜌
gives the step
)
(
2𝜌 𝑑
(5)
𝜂𝑘 = min 𝜂 max, 2
𝜎ˆ𝑘 + 𝜖
where 𝜖 > 0 prevents division by zero during the first iterations.2
The cap 𝜂 max keeps the discretization error under control in regions
where the gradient noise is small but the curvature is large.

3

Convergence Analysis

We assume that 𝑈 is 𝐿-smooth, that is, ∥∇𝑈 (𝑥)−∇𝑈 (𝑦) ∥ ≤ 𝐿∥𝑥 −𝑦 ∥
for all 𝑥, 𝑦, and that it is (𝑚, 𝑏)-dissipative, meaning ⟨∇𝑈 (𝑥), 𝑥⟩ ≥
𝑚∥𝑥 ∥ 2 − 𝑏. These conditions allow nonconvex potentials and are
standard in the analysis of Langevin algorithms. Under them, the
following bound holds for the law 𝜇𝑘 of the iterate 𝜃 𝑘 .
𝑊2 (𝜇𝑘 , 𝜋) ≤ 𝐶 1𝑒 −𝜆

Í

𝑗 <𝑘 𝜂 𝑗

√︁
√
𝑊2 (𝜇0, 𝜋) + 𝐶 2 𝜂 max𝑑 + 𝐶 3 max 𝜎ˆ 𝑗 𝜂 𝑗
𝑗 <𝑘

(6)
The constants 𝐶 1 , 𝐶 2 and 𝐶 3 depend on 𝐿, 𝑚 and 𝑏 but not on
the dimension, and 𝜆 is the contraction rate of the continuous
diffusion in the Wasserstein metric. The first term vanishes as
the accumulated step length grows, so the schedule converges
Í
provided that 𝑘 𝜂𝑘 = ∞, which holds because 𝜂𝑘 is bounded
below whenever 𝜎ˆ𝑘2 stays bounded.
2 In all experiments we use 𝜖 = 10 −8 and 𝜌 = 0.1; the results change by less than one

standard error for 𝜌 ∈ [0.05, 0.2] .

SLC ’26, November 2026, Fortaleza, Brazil

Duarte et al.

·10 −2

step size 𝜂𝑘

2

decreasing 𝜂 0 /(1 + 𝑘/𝜏 )
variance-aware

1.5

In both regimes the stochastic error is at most 2𝜌𝑑𝜂𝑘 , which
is a fraction 𝜌 of the injected noise. Summing over iterations and
applying the contraction of the diffusion yields the bound.

4
1
0.5
0

0

1,000

2,000

3,000

4,000

5,000

iteration 𝑘
Figure 1: Step sizes produced by the variance-aware schedule
and by a decreasing schedule with 𝜂 0 = 0.02 and 𝜏 = 400 during
the first 5000 iterations of Bayesian logistic regression.
The proof splits the error into a discretization part and a stochastic part. For the latter we write the minibatch error as a martingale
difference 𝜁𝑘 = 𝑔𝑘 − ∇𝑈 (𝜃 𝑘 ) and use the piecewise definition of the
step to separate the two regimes of the schedule:
(
𝜂 2 𝜎ˆ 2 , 𝜎ˆ𝑘2 ≤ 2𝜌𝑑/𝜂 max,
2
E∥𝜂𝑘 𝜁𝑘 ∥ ≤ max 𝑘
(7)
2𝜌𝑑 𝜂𝑘 , otherwise.

Experiments

We compare the schedule with a tuned constant step and with the
decreasing schedule 𝜂𝑘 = 𝜂 0 (1+𝑘/𝜏) −1 on two problems. The first is
Bayesian logistic regression on a data set with 𝑁 = 58,000 examples
and 𝑑 = 54 features. The second is a two-layer network with 256
hidden units, for which the posterior is multimodal. Accuracy is
measured by the kernel Stein discrepancy of 2000 retained samples
and by the predictive log-likelihood on held-out data.
On logistic regression the variance-aware schedule reaches a
Stein discrepancy of 0.031 ± 0.002, against 0.029 ± 0.002 for the best
constant step found by a grid of eight values and 0.047±0.004 for the
decreasing schedule. On the network, the predictive log-likelihood
is −0.412 for our schedule and −0.409 for the tuned constant step,
while the decreasing schedule stalls at −0.455 because its steps
become too small before the chain leaves the initial mode.

5

Conclusion

Coupling the step size of SGLD to an online estimate of the gradient
noise removes the most expensive tuning parameter of the method
at no cost in accuracy on the problems we tested. The analysis
extends to preconditioned variants in which 𝜎ˆ𝑘2 is replaced by a
diagonal estimate, a direction we leave for future work.

