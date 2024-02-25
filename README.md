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

| # | Date | Description | Slides |
|---|---|---|---|
| 1 | February, 7 | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive models (PixelCNN). | [slides](lectures/lecture1/Lecture1.pdf) |
|  |  | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Bayes theorem. |  |
| 2 | February, 14 | <b>Lecture 2:</b> Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear NF. Gaussian autoregressive NF. | [slides](lectures/lecture2/Lecture2.pdf) |
|  |  | <b>Seminar 2:</b> PixelCNN. |  |
| 3 | February, 21 | <b>Lecture 3:</b> Coupling layer (RealNVP). Continuous-in-time NF and neural ODE. Kolmogorov-Fokker-Planck equation for NF log-likelihood. FFJORD and Hutchinson's trace estimator. | [slides](lectures/lecture3/Lecture3.pdf) |
|  |  | <b>Seminar 3: Planar and Radial Flows. Forward vs Reverse KL. |  |
| 4 | February, 28 | <b>Lecture 4:</b> Adjoint method for continuous-in-time NF. Latent Variable Models (LVM). Variational lower bound (ELBO). | [slides](lectures/lecture4/Lecture4.pdf) |
|  |  | <b>Seminar 4:</b> RealNVP. Glow. |  |
| 5 | March, 6 | <b>Lecture 5:</b> Variational EM-algorithm. Amortized inference, ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). Uniform data dequantization. NF as VAE model. | [slides](lectures/lecture5/Lecture5.pdf) |
|  |  | <b>Seminar 5:</b> TBA. |  |
<!---
| 6 | March, 13 | <b>Lecture 6:</b> ELBO surgery and optimal VAE prior. NF-based VAE prior. Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). |  |
|  |  | <b>Seminar 6:</b> Planar Flow (coding), RealNVP. |  |
| 7 | March, 20 | <b>Lecture 7:</b> Gumbel-softmax trick (DALL-E). Likelihood-free learning. GAN optimality theorem.  |  |
|  |  | <b>Seminar 7:</b> Glow. |  |
| 8 | March, 27 | <b>Lecture 8:</b> Wasserstein distance. Wasserstein GAN (WGAN). WGAN with gradient penalty (WGAN-GP). Spectral Normalization GAN (SNGAN). |  |
|  |  | <b>Seminar 8:</b> Vanilla GAN in 1D coding. KL vs JS divergences. Mode collapse. Non-saturating GAN. |  |
| 9 | April, 3 | <b>Lecture 9:</b> f-divergence minimization. GAN evaluation. Inception score, FID, Precision-Recall, truncation trick. |  |
|  |  | <b>Seminar 9:</b> WGANs on multimodal 2D data. GANs zoo and evolution of GANs. StyleGAN coding. |  |
| 10 | April, 10 | <b>Lecture 10:</b>  |  |
|  |  | <b>Seminar 10:</b> StyleGAN: end discussions. Energy-Based models. |  |
| 11 | April, 17 | <b>Lecture 11:</b> Gaussian diffusion process. Gaussian diffusion model as VAE, derivation of ELBO. |  |
|  |  | <b>Seminar 11:</b> Gaussian diffusion process basics. |
| 12 | April, 24 | <b>Lecture 12:</b> Denoising diffusion probabilistic model (DDPM): reparametrization and overview. Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. |  |
|  |  | <b>Seminar 12:</b> Fast samplers: iDDPM and DDIM |  |
| 13 | May, 8 | <b>Lecture 13:</b> Score matching: implicit/sliced score matching, denoising score matching. Noise Conditioned Score Network (NCSN). DDPM vs NCSN. |  |
|  |  | <b>Seminar 13:</b> Noise Conditioned Score Network |  |
| 14 | May, 15 | <b>Lecture 14:</b> Variance Preserving and Variance Exploding SDEs. Model guidance: classifier guidance, classfier-free guidance. |  |
|  |  | <b>Seminar 14:</b> TBA |  |
-->

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | February, 14 | February, 28 | <ol><li>Theory (Kernel density estimation, alpha-divergences, curse of dimensionality).</li><li>PixelCNN (receptive field, autocomplete) on MNIST.</li><li>ImageGPT on MNIST.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-AIMasters-course/blob/main/homeworks/hw1.ipynb) |
<!---
| 2 | February, 28 | March, 13 | <ol><li>Theory (Sylvester flows, NF expressivity, Neural ODE Pontryagin theorem).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw2.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2024-DGM-AIMasters-course/blob/main/homeworks/hw2.ipynb) |
| 3 | March, 13 | March, 27 |  |  |
| 4 | March, 27 | April, 10 |  |  |
| 5 | April, 10 | April, 24 |  |  |
| 6 | April, 24 | May, 15 |  |  |
-->

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

