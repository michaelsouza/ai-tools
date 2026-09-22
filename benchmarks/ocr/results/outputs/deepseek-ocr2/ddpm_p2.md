image[[253, 88, 744, 145]]


figure_title[[307, 152, 690, 166]]
Figure 2: The directed graphical model considered in this work.

text[[171, 177, 825, 276]]
This paper presents progress in diffusion probabilistic models  \( [53] \) . A diffusion probabilistic model (which we will call a “diffusion model” for brevity) is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.

text[[171, 281, 826, 393]]
Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples. We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section  \( [4] \) ). In addition, we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling (Section  \( [3,2] \)   \( [55] \)   \( [61] \) ). We obtained our best sample quality results using this parameterization (Section  \( [4,2] \) ), so we consider this equivalence to be one of our primary contributions.

text[[171, 397, 826, 510]]
Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models (our models do, however, have log likelihoods better than the large estimates annealed importance sampling has been reported to produce for energy based models and score matching  \( [11] \)   \( [55] \) ). We find that the majority of our models' lossless codelengths are consumed to describe imperceptible image details (Section  \( [4,3] \) ). We present a more refined analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models.

sub_title[[173, 527, 311, 544]]
## 2 Background

text[[171, 557, 824, 614]]
Diffusion models [53] are latent variable models of the form  \(  p_{\theta}(\mathbf{x}_{0}) := \int p_{\theta}(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}  \) , where  \( x_{1}, \ldots, x_{T} \)  are latents of the same dimensionality as the data  \(  \mathbf{x}_{0} \sim q(\mathbf{x}_{0})  \) . The joint distribution  \(  p_{\theta}(\mathbf{x}_{0:T})  \)  is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at  \(  p(\mathbf{x}_{T}) = \mathcal{N}(\mathbf{x}_{T}; \mathbf{0}, \mathbf{I})  \) :

equation[[196, 621, 822, 654]]
 \[ p_{\theta}(\mathbf{x}_{0:T}):=p(\mathbf{x}_{T})\prod_{t=1}^{T}p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}),\qquad p_{\theta}(\mathbf{x}_{t-\mathbf{1}}|\mathbf{x}_{t}):=\mathcal{N}(\mathbf{x}_{t-\mathbf{i}};\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t},t),\boldsymbol{\Sigma}_{\theta}(\mathbf{x}_{t},\mathbf{t})) \quad (1) \] 

text[[172, 661, 824, 703]]
What distinguishes diffusion models from other types of latent variable models is that the approximate posterior  \( q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) \) , called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule  \( \beta_{1},\ldots,\beta_{T} \) :

equation[[243, 707, 822, 747]]
 \[ q(\mathbf{x}_{1:T}|\mathbf{x}_{0}):=\prod_{t=1}^{T}q(\mathbf{x}_{t}|\mathbf{x}_{t-1}),\qquad q(\mathbf{x}_{t}|\mathbf{\mathbf{x}}_{t-1}):=\mathcal{N}(\mathbf{x}_{t};\sqrt{1-\beta_{t}}\mathbf{x}_{t-1},\beta_{t}\mathbf{I}) \quad (2) \] 

text[[172, 756, 771, 771]]
Training is performed by optimizing the usual variational bound on negative log likelihood:

equation[[179, 775, 822, 810]]
 \[ \mathbb{E}\left[-\log p_{\theta}(\mathbf{x}_{0})\right]\leq\mathbb{E}_{q}\left[-\log\frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_{0})}\right]=\mathbb{E}_{q}\Bigg[-\log p(\mathbf{x}_{T})-\sum_{t\geq1}\log\frac{p_{\theta}(\mathbf{\mathbf{x}}_{t-1}|\mathbf{\mathbf{x}}_{t})}{q(\mathbf{\mathbf{x}}_{t}|\mathbf{\mathbf{x}}_{t-\mathbf{1}})}\Bigg]=:L \quad (3) \] 

text[[172, 815, 825, 888]]
The forward process variances  \( \beta_{t} \)  can be learned by reparameterization [33] or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of Gaussian conditionals in  \( p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t}) \) , because both processes have the same functional form when  \( \beta_{t} \)  are small [53]. A notable property of the forward process is that it admits sampling  \( x_{t} \)  at an arbitrary timestep t in closed form: using the notation  \( \alpha_{t} := 1 - \beta_{t} \)  and  \( \bar{\alpha}_{t} := \prod_{s=1}^{t} \alpha_{s} \) , we have

equation[[371, 891, 822, 907]]
 \[ q(\mathbf{x}_{t}|\mathbf{x}_{0})=\mathcal{N}(\mathbf{x}_{t};\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0},(1-\bar{\alpha}_{t})\mathbf{I}) \quad (4) \]