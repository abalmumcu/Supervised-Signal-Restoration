# Supervised-Signal-Restoration

The signal 𝐘=𝐇𝐗+𝐖 over an LTI causal channel given in the following figure. Here 𝐇 is a convolution matrix constituted by the system impulse response ℎ[𝑛] with length 5. The noise 𝐖∈ℝ256 is a white Gaussian vector with zero mean and variance 𝜎𝑤2. 𝐗 and 𝐖 are assumed to be uncorrelated. The signal 𝐗 is a Gaussian random vector with mean 𝛍𝑥 and covariance matrix 𝚺𝑥. 𝑀=254 input-output pairs {𝐱𝑖,𝐲𝑖} are given in the attached mat files. In the mat file, the 𝑖th column of matrix x and y correspond to 𝐱𝑖 and 𝐲𝑖, respectively.

![Screenshot_1](https://user-images.githubusercontent.com/71339227/130860268-2d9fd015-c8f6-4a01-9556-c1a06f29424d.png)

In this project; we applied 10-fold cross-validation to prepare the training and test samples, learn the parameters 𝛉={𝐇,𝛍𝑥,𝚺𝑥,𝜎𝑤2 }, find the MAP estimate of the signal 𝐱𝑗 and calculate the PSNR
