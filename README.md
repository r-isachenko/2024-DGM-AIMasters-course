# Deep Generative Models course, AIMasters, 2023

## Description
The course is devoted to modern generative models (mostly in the application to computer vision).

We will study the following types of generative models:
- autoregressive models,
- latent variable models,
- normalization flow models,
- adversarial models,
- diffusion models.

Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment.

The aim of the course is to introduce the student to widely used advanced methods of deep learning.

The course is accompanied by practical tasks that allow you to understand the principles of the considered models.

## Contact the author to join the course or for any other questions :)

- **telegram:** [@roman_isachenko](https://t.me/roman_isachenko)
- **e-mail:** roman.isachenko@phystech.edu

## Materials

| # | Date | Description | Slides | Video |
|---|---|---|---|---|
| 1 |  | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models (PixelCNN). |  | |
|  |  | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Kernel density estimation (KDE). |  |  |
| 2 |  | <b>Lecture 2:</b> Bayesian Framework. Latent Variable Models (LVM). Variational lower bound (ELBO). EM-algorithm, amortized inference. |  |  |
|  |  | <b>Seminar 2:</b> PixelCNN for MNIST and Binarized MNIST coding. |  |  |
| 3 |  | <b>Lecture 3:</b> ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). VAE limitations. Tighter ELBO (IWAE).  |  |  |
|  |  | <b>Seminar 3:</b> Latent Variable Models. Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm. |  |  |
| 4 |  | <b>Lecture 4:</b> Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear NF. Gaussian autoregressive NF. |  |  |
|  |  | <b>Seminar 4:</b> Variational EM algorithm for GMM. VAE: Implementation hints + Vanilla 2D VAE coding.  |  |  |
| 5 |  | <b>Lecture 5:</b> Coupling layer (RealNVP). NF as VAE model. Discrete data vs continuous model. Model discretization (PixelCNN++). Data dequantization: uniform and variational (Flow++). |  |  | |
|  |  | <b>Seminar 5:</b> VAE: posterior collapse, KL-annealing, free-bits. Normalizing flows: basics, planar flows, forward and backward kl for planar flows. |  |  |
| 6 |  | <b>Lecture 6:</b> ELBO surgery and optimal VAE prior. NF-based VAE prior. Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). |  |  |
|  |  | <b>Seminar 6:</b> Planar Flow (coding), RealNVP. |  |  |
| 7 |  | <b>Lecture 7:</b> Gumbel-softmax trick (DALL-E). Likelihood-free learning. GAN optimality theorem.  |  |  |
|  |  | <b>Seminar 7:</b> Glow. |  |  |
| 8 |  | <b>Lecture 8:</b> Wasserstein distance. Wasserstein GAN (WGAN). WGAN with gradient penalty (WGAN-GP). Spectral Normalization GAN (SNGAN). |  |  |
|  |  | <b>Seminar 8:</b> Vanilla GAN in 1D coding. KL vs JS divergences. Mode collapse. Non-saturating GAN. |  |  |
| 9 |  | <b>Lecture 9:</b> f-divergence minimization. GAN evaluation. Inception score, FID, Precision-Recall, truncation trick. |  |  |
|  |  | <b>Seminar 9:</b> WGANs on multimodal 2D data. GANs zoo and evolution of GANs. StyleGAN coding. |  |  |
| 10 |  | <b>Lecture 10:</b> Neural ODE. Adjoint method. Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). |  |  |
|  |  | <b>Seminar 10:</b> StyleGAN: end discussions. Energy-Based models. |  |  |
| 11 |  | <b>Lecture 11:</b> Gaussian diffusion process. Gaussian diffusion model as VAE, derivation of ELBO. |  |  |
|  |  | <b>Seminar 11:</b> Gaussian diffusion process basics. |  |
| 12 |  | <b>Lecture 12:</b> Denoising diffusion probabilistic model (DDPM): reparametrization and overview. Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. |  |  |
|  |  | <b>Seminar 12:</b> Fast samplers: iDDPM and DDIM |  |  |
| 13 |  | <b>Lecture 13:</b> Score matching: implicit/sliced score matching, denoising score matching. Noise Conditioned Score Network (NCSN). DDPM vs NCSN. |  |  |
|  |  | <b>Seminar 13:</b> Noise Conditioned Score Network |  |  |
| 14 |  | <b>Lecture 14:</b> Variance Preserving and Variance Exploding SDEs. Model guidance: classifier guidance, classfier-free guidance. |  |  |
|  |  | <b>Seminar 14:</b> TBA |  |  |

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| 6 |  |  |  |  |

## Game rules
- 6 homeworks each of 13 points = **78 points**
- oral cozy exam = **26 points**
- maximum points: 78 + 26 = **104 points**
### Final grade: `floor(relu(#points/8 - 2))`

## Prerequisities
- probability theory + statistics
- machine learning + basics of deep learning
- python + basics of one of DL frameworks (pytorch/tensorflow/etc)

## Previous episodes
- [2023, autumn, MIPT](https://github.com/r-isachenko/2023-DGM-MIPT-course)
- [2022-2023, autumn-spring, MIPT](https://github.com/r-isachenko/2022-2023-DGM-MIPT-course)
- [2022, autumn, AIMasters](https://github.com/r-isachenko/2022-2023-DGM-AIMasters-course)
- [2022, spring, OzonMasters](https://github.com/r-isachenko/2022-DGM-Ozon-course)
- [2021, autumn, MIPT](https://github.com/r-isachenko/2021-DGM-MIPT-course)
- [2021, spring, OzonMasters](https://github.com/r-isachenko/2021-DGM-Ozon-course)
- [2020, autumn, MIPT](https://github.com/r-isachenko/2020-DGM-MIPT-course)

