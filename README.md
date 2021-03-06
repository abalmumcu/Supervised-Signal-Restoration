# Supervised-Signal-Restoration

The signal π=ππ+π over an LTI causal channel given in the following figure. Here π is a convolution matrix constituted by the system impulse response β[π] with length 5. The noise πββ256 is a white Gaussian vector with zero mean and variance ππ€2. π and π are assumed to be uncorrelated. The signal π is a Gaussian random vector with mean ππ₯ and covariance matrix πΊπ₯. π=254 input-output pairs {π±π,π²π} are given in the attached mat files. In the mat file, the πth column of matrix x and y correspond to π±π and π²π, respectively.

![Screenshot_1](https://user-images.githubusercontent.com/71339227/130860268-2d9fd015-c8f6-4a01-9556-c1a06f29424d.png)

In this project; we applied 10-fold cross-validation to prepare the training and test samples, learn the parameters π={π,ππ₯,πΊπ₯,ππ€2 }, find the MAP estimate of the signal π±π and calculate the PSNR
