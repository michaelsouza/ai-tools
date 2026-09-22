Efficient training is therefore possible by optimizing random terms of $L$ with stochastic gradient descent. Further improvements come from variance reduction by rewriting $L$ (3) as:

 $$ \mathbb{E}_{q}\Bigg[\underbrace{D_{\mathrm{K L}}(q(\mathbf{x}_{T}|\mathbf{x}_{0})\parallel p(\mathbf{x}_{T}))}_{L_{T}}+\sum_{t>1}\underbrace{D_{\mathrm{K L}}(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})\parallel p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}))}_{L_{t-1}}\underbrace{-\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{1})}_{L_{0}}\Bigg] $$ 

(See Appendix A for details. The labels on the terms are used in Section 3) Equation (5) uses KL divergence to directly compare  $ p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) $ against forward process posteriors, which are tractable when conditioned on  $ \mathbf{x}_{0} $:

 $$ \begin{array}{r}{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})=\mathcal{N}(\mathbf{x}_{t-1};\tilde{\mathbf{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{0}),\tilde{\boldsymbol{\beta}}_{t}\mathbf{I}),}\end{array} $$ 

 $$ \mathrm{where}\quad\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{0}):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}\mathbf{x}_{0}+\frac{\sqrt{\alpha_{t}}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_{t}}\mathbf{x}_{t}\quad\mathrm{and}\quad\tilde{\boldsymbol{\beta}}_{t}:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\boldsymbol{\beta}_{t} $$ 

Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

## 3 Diffusion models and denoising autoencoders

Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances  $ \beta_t $ of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section  $ [3.2] $) that leads to a simplified, weighted variational bound objective for diffusion models (Section  $ [3.4] $). Ultimately, our model design is justified by simplicity and empirical results (Section  $ [4] $). Our discussion is categorized by the terms of Eq. (5).

### 3.1 Forward process and  $ L_{T} $

We ignore the fact that the forward process variances  $ \beta_{t} $ are learnable by reparameterization and instead fix them to constants (see Section 4 for details). Thus, in our implementation, the approximate posterior  $ q $ has no learnable parameters, so  $ L_{T} $ is a constant during training and can be ignored.

### 3.2 Reverse process and  $ L_{1:T-1} $

Now we discuss our choices in $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})=\mathcal{N}(\mathbf{x}_{t-1};\boldsymbol{\mu}_{\theta}(x_{t},t),\boldsymbol{\Sigma}_{\theta}(x_{t},t))$ for $1<t\leq T$. First, we set $\boldsymbol{\Sigma}_{\theta}(\mathbf{x}_{t},t)=\sigma_{t}^{2}\mathbf{I}$ to untrained time dependent constants. Experimentally, both $\sigma_{t}^{2}=\beta_{t}$ and $\sigma_{t}^{2}=\beta_{t}=\frac{1-\alpha_{t-1}}{1-\alpha_{t}}\beta_{t}$ had similar results. The first choice is optimal for $\mathbf{x}_{0}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, and the second is optimal for $\mathbf{x}_{0}$ deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance [53].

Second, to represent the mean $\mu_{\theta}(\mathbf{x}_t, t)$, we propose a specific parameterization motivated by the following analysis of $L_t$. With $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_{\theta}(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$, we can write:

 $$ {\cal L}_{t-1}=\mathbb{E}_{q}\left[\frac{1}{2\sigma_{t}^{2}}\|\tilde{\mu}_{t}(\mathbf{x}_{t},\mathbf{x}_{0})-\mu_{\theta}(\mathbf{x}_{t},t)\|^{2}\right]+C $$ 

where $C$ is a constant that does not depend on $\theta$. So, we see that the most straightforward parameterization of $\mu_{\theta}$ is a model that predicts $\tilde{\mu}_{t}$, the forward process posterior mean. However, we can expand Eq. (8) further by reparameterizing Eq. (4) as $\mathbf{x}_{t}(\mathbf{x}_{0}, \epsilon) = \sqrt{\alpha_{t}} \mathbf{x}_{0} + \sqrt{1 - \tilde{\alpha}_{t}} \epsilon$ for $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and applying the forward process posterior formula (7):

 $$ \begin{aligned}L_{t-1}-C&=\mathbb{E}_{\mathbf{x}_{0},\epsilon}\Bigg[\frac{1}{2\sigma_{t}^{2}}\left\|\tilde{\mu}_{t}\bigg(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon),\frac{1}{\sqrt{\alpha_{t}}}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon)-\sqrt{1-\bar{\alpha}_{t}}\epsilon)\bigg)-\mu_{\theta}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon),t)\right\|^{2}\Bigg]\\&=\mathbb{E}_{\mathbf{x}_{0},\epsilon}\Bigg[\frac{1}{2\sigma_{t}^{2}}\left\|\frac{1}{\sqrt{\alpha_{t}}}\bigg(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon)-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon\bigg)-\mu_{\theta}(\mathbf{x}_{t}(\mathbf{x}_{0},\epsilon),t)\right\|^{2}\Bigg]\\ \end{aligned} $$ 