# TF-32-Tensor-Test-Suite
Tensorflow and Pytorch test for tensor cores performance. Tested with 2.8.0 and 1.11.0 respectively.

Tests were written after Cuda 11.6.2 and Cudnn 8.4.0.27 with Debian-like OS.


There are some parameters to be setup before test run ↓

# Iterations number
*-i* or *--iters*
Default value - 100

# Matrix size n x n
*-s* or *--size*
Default value - 8192

# Tensor cores use selector 
*-t* *--tensor*
Default value - True

It is recommended not to change the values of *iters* and *size* without specific need.

*tensor* allows to switch between TF-32 math when selected and FP-32 when not selected.

# Other parameters

If you need more information while running the test use *-v* or *--verbose* to make it more talkative.

For Test-Suite there two more parameters: *-a* to run all tests and *-r* or *--runs* to override predifined 13 runs.


# Performance boosts for TF-32 math
For single GeForce RTX 3060 average measured performance acceleration medians were **37,02%** with TF-32 math for Tensorflow and **81,07%** for PyTorch. Tests iters were 100, 1'000, 10'000, 50'000 and 100'000 with n = 8192.

**3060 tests **

For single GeForce RTX 3060 set of 13 runs with exlude of 1st one with iter = 100 medians for TensorFlow2 were 9594,22 Gflops/s FP-32 (σ 5,5555) and 13102,84 Gflop/s TF-32 (σ 7,0173), acceleration **36,57%**.

For single GeForce RTX 3060 set of 13 runs with exlude of 1st one with iter = 100 medians for PyTorch were 7255,66 Glop/s FP-32 (σ 4,7315) and 13137,63 Gflop/s TF-32 (σ 23,8104), acceleration **81,07%**.

**3080 TI tests **

For single GeForce RTX 3080 TI set of 13 runs with exlude of 1st one with iter = 100 medians for TensorFlow2 were 25481,71,22 Gflops/s FP-32 (σ 43,8388) and 39424,27,84 Gflop/s TF-32 (σ 1,3126), acceleration **54,72%**.

For single GeForce RTX 3080 TI set of 13 runs with exlude of 1st one with iter = 100 medians for PyTorch were 21671,57 Glop/s FP-32 (σ 31,9975) and 38914,05 Gflop/s TF-32 (σ 71,7127), acceleration **79,56%**.

