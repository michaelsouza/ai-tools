Figure 2: The directed graphical model considered in this work.(259,90),(744,145)

This paper presents progress in diffusion probabilistic models [53]. A diffusion probabilistic model (which we will call a “diffusion model” for brevity) is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.

Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples. We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section 4). In addition, we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling (Section 3.2) [55, 61]. We obtained our best sample quality results using this parameterization (Section 4.2), so we consider this equivalence to be one of our primary contributions.

Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models (our models do, however, have log likelihoods better than the large estimates annealed importance sampling has been reported to produce for energy based models and score matching [11, 55]). We find that the majority of our models' lossless codelengths are consumed to describe imperceptible image details (Section 4.3). We present a more refined analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models.

2 Background

Diffusion models [53] are latent variable models of the form $ p_{\theta}(x_{0}) \mathrel{\text{:=}} \int p_{\theta}(x_{0:T}) \, dx_{1:T} $, where $ x_{1}, \ldots, x_{T} $ are latents of the same dimensionality as the data $ x_{0} \sim q(x_{0}) $. The joint distribution $ p_{\theta}(x_{0:T}) $ is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at $ p(x_{T}) = \mathcal{N}(x_{T}; 0, \mathbf{I}) $:

$$ p_{\theta}(x_{0:T}) \mathrel{\text{:=}} p(x_{T}) \prod_{t=1}^{T} p_{\theta}(x_{t-1} | x_{t}), \qquad p_{\theta}(x_{t-1} | x_{t}) \mathrel{\text{:=}} \mathcal{N}(x_{t-1}; \boldsymbol{\mu}_{\theta}(x_{t}, t), \boldsymbol{\Sigma}_{\theta}(x_{t}, t)) \tag{1} $$

What distinguishes diffusion models from other types of latent variable models is that the approximate posterior $ q(x_{1:T} | x_{0}) $, called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $ \beta_{1}, \ldots, \beta_{T} $:

$$ q(x_{1:T} | x_{0}) \mathrel{\text{:=}} \prod_{t=1}^{T} q(x_{t} | x_{t-1}), \qquad q(x_{t} | x_{t-1}) \mathrel{\text{:=}} \mathcal{N}(x_{t}; \sqrt{1-\beta_{t}} x_{t-1}, \beta_{t} \mathbf{I}) \tag{2} $$

Training is performed by optimizing the usual variational bound on negative log likelihood:

$$ \mathbb{E}\left[-\log p_{\theta}(x_{0})\right] \leq \mathbb{E}_{q}\left[-\log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_{0})}\right]=\mathbb{E}_{q}\left[-\log p(x_{T})-\sum_{t\geq 1}\log\frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1})}\right]=:L \tag{3} $$

The forward process variances $ \beta_{t} $ can be learned by reparameterization [33] or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of Gaussian conditionals in $ p_{\theta}(x_{t-1} | x_{t}) $, because both processes have the same functional form when $ \beta_{t} $ are small [53]. A notable property of the forward process is that it admits sampling $ x_{t} $ at an arbitrary timestep $ t $ in closed form: using the notation $ \alpha_{t} \mathrel{\text{:=}} 1-\beta_{t} $ and $ \bar{\alpha}_{t} \mathrel{\text{:=}} \prod_{s=1}^{t} \alpha_{s} $, we have

$$ q(x_{t} | x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar{\alpha}_{t}} x_{0}, (1-\bar{\alpha}_{t}) \mathbf{I}) \tag{4} $$