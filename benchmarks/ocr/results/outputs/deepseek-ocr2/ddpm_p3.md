text[[171, 91, 824, 121]]
Efficient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L (3) as:

equation[[183, 127, 822, 167]]
 \[ \mathbb{E}_{q}\Biggl[\underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{T}|\mathbf{x}_{0})\parallel p(\mathbf{x}_{T}))}_{L_{T}}+\sum_{t>1}D_{\mathrm{KL}}(q(\underbrace{\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0}}_{L_{t-1}})\parallel p_{\theta}(\underbrace{\mathbf{x}_{t-\mathbf{1}}|\mathbf{x}_{t}}_{L_{0}}))-\underbrace{\log p_{\theta}(\mathbf{x}_{0}|\mathbf{x}_{\mathbf{1}})}_{L_{0}}\Biggr] \quad (5) \] 

text[[171, 171, 824, 214]]
(See Appendix A for details. The labels on the terms are used in Section \( ^{[3]} \) ) Equation (5) uses KL divergence to directly compare  \( p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \)  against forward process posteriors, which are tractable when conditioned on  \( x_{0} \) :

equation[[249, 218, 822, 235]]
 \[ q(\mathbf{x}_{t-1}\vert\mathbf{x}_{t},\mathbf{x}_{0})=\mathcal{N}(\mathbf{x}_{t-1};\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{{0}}),\tilde{\beta}_{t}\mathbf{I}), \quad (6) \] 

equation[[223, 238, 822, 269]]
 \[  where\quad\tilde{\boldsymbol{\mu}}_{t}(\boldsymbol{x}_{t},\boldsymbol{x}_{0}):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}\boldsymbol{x}_{0}+\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t}\quad and\quad\tilde{\beta}_{t}:=\frac{1-\bar{\alpha}_{t-\mathbf{1}}}{1-\bar{\alpha}_{t}}\beta_{t} \quad (7) \] 

text[[171, 273, 824, 315]]
Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

sub_title[[172, 334, 584, 350]]
## 3 Diffusion models and denoising autoencoders

text[[170, 364, 825, 463]]
Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances  \( \beta_{t} \)  of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section \( ^{[3,2]} \) ) that leads to a simplified, weighted variational bound objective for diffusion models (Section \( ^{[3.4]} \) ). Ultimately, our model design is justified by simplicity and empirical results (Section \( ^{[4]} \) ). Our discussion is categorized by the terms of Eq. (5).

sub_title[[172, 477, 385, 492]]
## 3.1 Forward process and  \( L_{T} \) 

text[[171, 503, 824, 546]]
We ignore the fact that the forward process variances  \( \beta_{t} \)  are learnable by reparameterization and instead fix them to constants (see Section \( ^{[4]} \)  for details). Thus, in our implementation, the approximate posterior q has no learnable parameters, so  \( L_{T} \)  is a constant during training and can be ignored.

sub_title[[172, 561, 405, 576]]
## 3.2 Reverse process and  \( L_{1:T-1} \) 

text[[171, 586, 825, 674]]
Now we discuss our choices in  \(  p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) = \mathcal{N}(\mathbf{x}_{t-\mathbf{1}}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_{t, t}))  \)  for  \( 1 < t \leq T \) . First, we set  \(  \boldsymbol{\Sigma}_{\theta} (\mathbf{x}_{t}, t) = \sigma_{t}^{2} \mathbf{I}  \)  to untrained time dependent constants. Experimentally, both  \(  \sigma_{t}^{2} = \beta_{t}  \)  and  \(  \sigma_{\ell}^{2} = \tilde{\beta}_{\ell} = \frac{1 - \alpha_{\ell - 1}}{\alpha_{\ell}} \beta_{\ell}  \)  had similar results. The first choice is optimal for  \(  x_{0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})  \) , and the second is optimal for  \( x_{0} \)  deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance [53].

text[[171, 679, 824, 708]]
Second, to represent the mean  \( \boldsymbol{\mu}_{\theta}(\mathbf{x}_{t},t) \) , we propose a specific parameterization motivated by the following analysis of  \( L_{t} \) . With  \( p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})=\mathcal{N}(\mathbf{x}_{t-\mathbf{1}};\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t},t),\sigma_{t}^{2}\mathbf{I}) \) , we can write:

equation[[335, 713, 822, 744]]
 \[ L_{t-1}=\mathbb{E}_{q}\left[\frac{1}{2\sigma_{t}^{2}}\|\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{0})-\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t},t)\|^{2}\right]+C \quad (8) \] 

text[[171, 749, 824, 807]]
where C is a constant that does not depend on  \( \theta \) . So, we see that the most straightforward parameterization of  \( \mu_{\theta} \)  is a model that predicts  \( \tilde{\mu}_{t} \) , the forward process posterior mean. However, we can expand Eq. (8) further by reparameterizing Eq. (4) as  \( \mathbf{x}_{t}(\mathbf{x}_{0},\mathbf{\epsilon})=\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon} \)  for  \( \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}) \)  and applying the forward process posterior formula (7):

equation[[182, 810, 822, 863]]
 \[ L_{t-1}-C=\mathbb{E}_{\mathbf{x}_{0},\boldsymbol{\epsilon}}\Biggl[\frac{1}{2\sigma_{t}^{2}}\left\|\tilde{\boldsymbol{\mu}}_{t}\Big(\mathbf{x}_{t}(\mathbf{x}_{0},\boldsymbol{\varepsilon}),\frac{1}{\sqrt{\bar{\alpha}_{t}}}(\mathbf{x}_{t}(\mathbf{x}_{\mathbf{0}},\boldsymbol{\varepsilon})-\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\varepsilon})\Big)-\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}(\mathbf{\mathbf{x}}_{0},\boldsymbol{\varepsilon}),t)\right\|^{2}\Biggr] \quad (9) \] 

equation[[252, 867, 822, 907]]
 \[ =\mathbb{E}_{\mathbf{x}_{0},\boldsymbol{\epsilon}}\Biggl[\frac{1}{2\sigma_{t}^{2}}\left\|\frac{1}{\sqrt{\bar{\alpha}_{t}}}\left(\mathbf{x}_{t}(\mathbf{x}_{0},\boldsymbol{\varepsilon})-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon}}\boldsymbol{\varepsilon}\right)-\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}(\mathbf{\mathbf{x}}_{0},\boldsymbol{\varepsilon}),t)\right\|^{2}\Biggr] \quad (10) \]