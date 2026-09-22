text[[170, 93, 812, 125]]
Efficient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L  \( \textcircled{3} \)  as:

equation[[186, 130, 811, 170]]
 \[ \mathbb{E}_{q}\Biggl[\underbrace{D_{\mathrm{K L}}(q(\mathbf{x}_{T}|\mathbf{x}_{0})\parallel p(\mathbf{x}_{T}))}_{L_{T}}+\sum_{t>1}\underbrace{D_{\mathrm{K L}}\big(q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})\parallel\underbrace{p_{\theta}(\mathbf{x}_{t-1}\mid\mathbf{x}_{t})}_{L_{t-1})}-\underbrace{\log p_{\theta}(\mathbf{x}_{0}\mid\mathbf{x}_{1})}_{L_{0}}\Biggr] \quad (5) \] 

text[[172, 173, 814, 218]]
(See Appendix A for details. The labels on the terms are used in Section \( ^{[3]} \) ) Equation  \( \textcircled{5} \)  uses KL divergence to directly compare  \( p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \)  against forward process posteriors, which are tractable when conditioned on  \( x_{0} \) :

equation[[250, 218, 813, 238]]
 \[ q(\mathbf{x}_{t-1}\mid\mathbf{x}_{t},\mathbf{x}_{0})=\mathcal{N}(\mathbf{x}_{t-1};\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{{0}}),\tilde{\beta}_{t}\mathbf{I}), \quad (6) \] 

equation[[223, 238, 813, 272]]
 \[ \mathrm{w h e r e}\quad\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{0}):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}\mathbf{x}_{0}+\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}{\mathbf{x}}_{t}\quad\mathrm{a n d}\quad\tilde{\beta}_{t}:=\frac{1-\bar{\alpha}_{t\!-\!1}}{1-\bar{\alpha}_{t}}\beta_{t} \quad (7) \] 

text[[174, 274, 816, 319]]
Consequently, all KL divergences in Eq.  \( \textcircled{5} \)  are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance Monte Carlo estimates.

sub_title[[175, 336, 578, 354]]
## 3 Diffusion models and denoising autoencoders

text[[175, 364, 818, 465]]
Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances  \( \beta_{t} \)  of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section \( ^{[3,2]} \) ) that leads to a simplified, weighted variational bound objective for diffusion models (Section \( ^{[3.4]} \) ). Ultimately, our model design is justified by simplicity and empirical results (Section \( ^{[4]} \) ). Our discussion is categorized by the terms of Eq.  \( \textcircled{5} \) .

sub_title[[178, 479, 385, 494]]
## 3.1 Forward process and  \( L_{T} \) 

text[[177, 501, 819, 547]]
We ignore the fact that the forward process variances  \( \beta_{t} \)  are learnable by reparameterization and instead fix them to constants (see Section \( ^{[4]} \)  for details). Thus, in our implementation, the approximate posterior q has no learnable parameters, so  \( L_{T} \)  is a constant during training and can be ignored.

sub_title[[179, 562, 406, 577]]
## 3.2 Reverse process and  \( L_{1:T-1} \) 

text[[178, 583, 821, 675]]
Now we discuss our choices in  \(  p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_{t, t}))  \)  for  \( 1 < t \leq T \) . First, we set  \(  \boldsymbol{\Sigma}_{\theta} (\mathbf{x}_{t}, t) = \sigma_{t}^{2} \mathbf{I}  \)  to untrained time dependent constants. Experimentally, both  \(  \sigma_{t}^{2} = \beta_{t}  \)  and  \(  \sigma_{t}^{\tilde{\beta}} = \tilde{\beta}_{t} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha_{t}}} \beta_{t}  \) , had similar results. The first choice is optimal for  \(  \mathbf{x}_{0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})  \) , and the second is optimal for  \( x_{0} \)  deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance [53].

text[[180, 676, 821, 708]]
Second, to represent the mean  \( \mu_{\theta}(\mathbf{x}_{t},t) \) , we propose a specific parameterization motivated by the following analysis of  \( L_{t} \) . With  \( p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})=\mathcal{N}(\mathbf{x}_{t-1};\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t},t),\sigma_{t}^{2}\mathbf{I}) \) , we can write:

equation[[338, 710, 821, 742]]
 \[ L_{t-1}=\mathbb{E}_{q}\left[\frac{1}{2\sigma_{t}^{2}}\|\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t},\mathbf{x}_{0})-\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t},t)\|^{2}\right]+C \quad (8) \] 

text[[181, 745, 823, 805]]
where C is a constant that does not depend on  \( \theta \) . So, we see that the most straightforward parameterization of  \( \mu_{\theta} \)  is a model that predicts  \( \tilde{\mu}_{t} \) , the forward process posterior mean. However, we can expand Eq. (8) further by reparameterizing Eq. (4) as  \( \mathbf{x}_{t}(\mathbf{x}_{0},\mathbf{\epsilon})=\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon \)  for  \( \epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}) \)  and applying the forward process posterior formula (7):

equation[[193, 806, 823, 857]]
 \[ L_{t-1}-C=\mathbb{E}_{\mathbf{x}_{0},\epsilon}\Biggl[\frac{1}{2\sigma_{t}^{2}}\left\|\tilde{\boldsymbol{\mu}}_{t}\Biggl(\mathbf{x}_{t}(\mathbf{x}_{0},\boldsymbol{\epsilon}),\frac{1}{\sqrt{\bar{\alpha}_{t}}}(\mathbf{x}_{t}(\mathbf{x_{0}},\boldsymbol{\epsilon})-\sqrt{1-\bar{\alpha}_{t}}\boldsymbol{\epsilon})\Biggr)-\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}(\mathbf{\mathbf{x}_{0}},\boldsymbol{\epsilon}),t)\right\|^{2}\Biggr] \quad (9) \] 

equation[[263, 862, 823, 904]]
 \[ =\mathbb{E}_{\mathbf{x}_{0},\epsilon}\Biggl[\frac{1}{2\sigma_{t}^{2}}\Bigg\|\frac{1}{\sqrt{\alpha_{t}}}\Bigg(\mathbf{x}_{t}(\mathbf{x}_{0},\boldsymbol{\epsilon})-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\boldsymbol{\epsilon}\Bigg)-\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}(\mathbf{\mathbf{x}_{0}},\boldsymbol{\epsilon}),t)\Bigg\|^{2}\Biggr] \quad (10) \]