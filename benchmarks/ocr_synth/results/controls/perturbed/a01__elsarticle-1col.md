# Variance-Aware Step Sizes for Stochastic Gradient Langevin Dynamics on Nonconvex Objectives

Helena M. Duarte, Rafael T. Okafor, Sun-Hee Park

Department of Applied Mathematics, Northgate Institute of Technology

**Abstract.** Stochastic gradient Langevin dynamics (SGLD) samples from a posterior by adding Gaussian noise of variance $2\eta_k$ to each stochastic gradient step, but its accuracy depends on a step size $\eta_k$ that is usually tuned by hand. We propose a schedule that couples $\eta_k$ to a running estimate $\hat{\sigma}_k^2$ of the gradient noise, so that the injected noise dominates the minibatch noise at every iteration. For targets $\pi(\theta) \propto e^{-U(\theta)}$ with $L$-smooth potentials satisfying a dissipativity condition, we bound the 2-Wasserstein distance between the law of the iterates and $\pi$ by $O(\sqrt{\eta_k}\, d^{1/2} + \hat{\sigma}_k \eta_k^{1/2})$. Experiments on Bayesian logistic regression and a two-layer network show that the schedule matches a tuned constant step while removing the search over $\eta$.

**Keywords:** Langevin dynamics, Bayesian inference, step-size adaptation, Wasserstein distance

## 1. Introduction

SGLD replaces the full gradient by an unbiased minibatch estimate and drops the Metropolis correction.[^1]The update reads

Bayesian inference for large models requires samples from a posterior whose density is known only up to a normalizing constant. Writing the target as $\pi(\theta) \propto \exp(-U(\theta))$ with $\theta \in \mathbb{R}^d$, the overdamped Langevin diffusion $\mathrm{d}\theta_t = -\nabla U(\theta_t)\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t$ has $\pi$ as its stationary law, and its Euler–Maruyama discretization gives the unadjusted Langevin algorithm.

Journal of Computational Statistics and Learning 12 (2026) 101104

 When $U$ is a sum over $N$ data points, each gradient costs $O(N)$ operations, which is prohibitive for the data sets common in machine learning.

$$
\theta_{k+1} = \theta_k - \eta_k \widehat{\nabla U}(\theta_k) + \sqrt{2\beta_k}\, \xi_k, \qquad \xi_k \sim \mathcal{N}(0, I_d)
$$

where $\widehat{\nabla U}(\theta_k) = \frac{N}{n} \sum_{i \in S_k} \nabla U_i(\theta_k)$ is computed on a minibatch $S_k$ of size $n$. Two sources of error compete in this scheme. The discretization error grows with $\eta_k$, while the minibatch noise has covariance of order $\eta_k^2$ and becomes negligible relative to the injected noise only when $\eta_k$ is small. Practitioners therefore decrease $\eta_k$ slowly or tune a constant step, and both choices are made without information about the actual noise level of the gradient estimator.

## 2. Variance-Aware Schedule

Our schedule estimates the trace of the gradient covariance with an exponential moving average and shrinks the step whenever the minibatch noise would exceed a fixed fraction $\rho \in (0, 1)$ of the injected noise. Let $g_k = \widehat{\nabla U}(\theta_k)$ and define the recursions

$$
\begin{aligned}
m_k &= \beta\, m_{k-1} + (1 - \beta)\, g_k, \\
v_k &= \beta\, v_{k-1} + (1 - \beta)\, \lVert g_k - m_{k-1} \rVert^2, \\
\hat{\sigma}_k^2 &= \frac{v_k}{1 - \beta^k}
\end{aligned}
$$

with $m_0 = 0$, $v_0 = 0$ and $\beta \in [0.9, 0.999)$. The minibatch noise injected per step has variance $\eta_k^2 \hat{\sigma}_k^2$, whereas the Gaussian term contributes $2\eta_k d$. Requiring the ratio of the two to stay below $\rho$ gives the step

$$
\eta_k = \min\left\{ \eta_{\max},\; \frac{2 \rho\, d}{\hat{\sigma}_k^2 + \epsilon} \right\}
$$

where $\epsilon > 0$ prevents division by zero during the first iterations.[^2]The cap $\eta_{\max}$ keeps the discretization error under control in regions where the gradient noise is small but the curvature is large.

Figure 1: Step sizes produced by the variance-aware schedule and by a decreasing schedule with $\eta_0 = 0.02$ and $\tau = 400$ during the first 5000 iterations of Bayesian logistic regression.

## 3. Convergence Analysis

We assume that $U$ is $L$-smooth, that is, $\lVert \nabla U(x) - \nabla U(y) \rVert \le L \lVert x - y \rVert$ for all $x, y$, and that it is $(m, b)$-dissipative, meaning $\langle \nabla U(x), x \rangle \ge m \lVert x \rVert^2 - b$. These conditions allow nonconvex potentials and are standard in the analysis of Langevin algorithms. Under them, the following bound holds for the law $\mu_k$ of the iterate $\theta_k$.

$$
W_2(\mu_k, \pi) \le C_1 e^{-\lambda \sum_{j<k} \eta_j} W_2(\mu_0, \pi) + C_2 \sqrt{\eta_{\max} d} + C_3 \max_{j<k} \hat{\sigma}_j \sqrt{\eta_j}
$$

The constants $C_1$, $C_2$ and $C_3$ depend on $L$, $m$ and $b$ but not on the dimension, and $\lambda$ is the contraction rate of the continuous diffusion in the Wasserstein metric. The first term vanishes as the accumulated step length grows, so the schedule converges provided that $\sum_k \eta_k = \infty$, which holds because $\eta_k$ is bounded below whenever $\hat{\sigma}_k^2$ stays bounded.

The proof splits the error into a discretization part and a stochastic part. For the latter we write the minibatch error as a martingale difference $\zeta_k = g_k - \nabla U(\theta_k)$ and use the piecewise definition of the step to separate the two regimes of the schedule:

$$
\mathbb{E}\lVert \eta_k \zeta_k \rVert^2 \le \begin{cases} \eta_{\max}^2 \hat{\sigma}_k^2, & \hat{\sigma}_k^2 \le 2\rho d / \eta_{\max}, \\ 2\rho d\, \eta_k, & \text{otherwise}. \end{cases}
$$

In both regimes the stochastic error is at most $2\rho d \eta_k$, which is a fraction $\rho$ of the injected noise. Summing over iterations and applying the contraction of the diffusion yields the bound.

## 4. Experiments

We compare the schedule with a tuned constant step and with the decreasing schedule $\eta_k = \eta_0 (1 + k/\tau)^{-1}$ on two problems. The first is Bayesian logistic regression on a data set with $N = 58,000$ examples and $d = 54$ features. The second is a two-layer network with $256$ hidden units, for which the posterior is multimodal. Accuracy is measured by the kernel Stein discrepancy of $2000$ retained samples and by the predictive log-likelihood on held-out data.

On logistic regression the variance-aware schedule reaches a Stein discrepancy of $0.031 \pm 0.002$, against $0.029 \pm 0.002$ for the best constant step found by a grid of eight values and $0.047 \pm 0.004$ for the decreasing schedule. On the network, the predictive log-likelihood is $-0.412$ for our schedule and $-0.409$ for the tuned constant step, while the decreasing schedule stalls at $-0.455$ because its steps become too small before the chain leaves the initial mode.

## 5. Conclusion

Coupling the step size of SGLD to an online estimate of the gradient noise removes the most expensive tuning parameter of the method at no cost in accuracy on the problems we tested. The analysis extends to preconditioned variants in which $\hat{\sigma}_k^2$ is replaced by a diagonal estimate, a direction we leave for future work.

[^2]: In all experiments we use $\epsilon = 10^{-8}$ and $\rho = 0.1$; the results change by less than one standard error for $\rho \in [0.05, 0.2]$.
