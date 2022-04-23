# TF-32-Tensor-Test-Suite
Tensorflow and Pytorch test for tensor cores performance. Tested with 2.8.0 and 1.11.0 respectivly.

There are three values to be setup before test run

# Iterations number
iters = 100

# Matrix size n x n
n = 8192

# Tensor cores use selector 
use_tensor_cores = True

It is recommended not to change the values of **iters** and **n** without specific need.

**use_tensor_cores** allows to switch between TF-32 math when set to **True** and FP-32 math when set to **False**
