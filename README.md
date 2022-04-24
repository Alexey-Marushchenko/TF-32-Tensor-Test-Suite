# TF-32-Tensor-Test-Suite
Tensorflow and Pytorch test for tensor cores performance. Tested with 2.8.0 and 1.11.0 respectively.

Tests were written after Cuda 11.6.2 and Cudnn 8.4.0.27 with Debian-like OS.

There are three values to be setup before test run ↓

# Iterations number
**iters**

Default value - 100

# Matrix size n x n
**n**

Default value - 8192

# Tensor cores use selector 
**use_tensor_cores**

Default value - True

It is recommended not to change the values of **iters** and **n** without specific need.

**use_tensor_cores** allows to switch between TF-32 math when set to **True** and FP-32 math when set to **False**

# Performance boosts for TF-32 math
For single NVIDIA GeForce RTX 3060 average measured performance acceleration medians were **37,02%** with TF-32 math for Tensorflow and **81,08%** for PyTorch. Tests iters were 100, 1'000, 10'000, 50'000 and 100'000 with n = 8192.

For set of 13 runs with exlude of 1st on with iter = 100 acceleration medians were **36,57%** with TF-32 math for Tensorflow and **81,07%** for PyTorch.
