#
# Copyright (C) 2022 Alexey Marushchenko <alexey@marush.com>
#
# Based on https://pytorch.org/docs/stable/notes/cuda.html
#
# This file is part of TF-32 Tensor Test Suite.
#
# This is free software: you can redistribute it and/or modifympere
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Version 20220423_1008
# Target torch version 1.11.0
# Target CUDA version 11.6.2
# Target CUDNN version 8.4.0

import os
import sys
import time
import torch

dtype = torch.float32

# Iterations number
iters = 100

# Matrix size n x n
n = 8192

# Tensor cores use selector
use_tensor_cores = True

cuda = torch.device('cuda:0')
g_cuda=torch.Generator(cuda)
g_cuda.seed()

in_matrix = torch.rand(n, n, generator=g_cuda, dtype=dtype, device=cuda)
ret_matrix = torch.rand(n, n, generator=g_cuda, dtype=dtype, device=cuda)

torch.backends.cuda.matmul.allow_tf32 = use_tensor_cores
torch.backends.cudnn.allow_tf32 = use_tensor_cores
print("\nIs TF32 matmul enabled: " + str(torch.backends.cuda.matmul.allow_tf32))
print("Is TF32 cudnn enabled: " + str(torch.backends.cudnn.allow_tf32))

# Warmup with at least 1200 iterations
print('\nKeep calm while warming up the card...\n')
for i in range(1200):
   ret_matrix = in_matrix @ ret_matrix

start_time = time.time()
iter_time = start_time
 
for i in range(1, iters+1):
   ret_matrix = in_matrix @ ret_matrix

   if (i/iters*100).is_integer():
      tran_time = time.time()
      print(str(time.ctime()) + ' | Iter num %d of %d | %.2f ' % (i, iters, i/iters*100) + r"%" + ' done | Est. time remain is %.2f sec.' % ((tran_time - iter_time)*(1-i/iters)*100), flush=True)
      iter_time=tran_time

end_time = time.time()

ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed_time = (end_time - start_time)
rate = iters*ops/elapsed_time/10**9

print("\nWas TF32 matmul enabled: " + str(torch.backends.cuda.matmul.allow_tf32))
print("Was TF32 cudnn enabled: " + str(torch.backends.cudnn.allow_tf32))
print('\n%d x %d matmul took: %.2f sec per iter at rate %.2f G ops/sec. Total time is: %.2f sec' % (n, n, elapsed_time/iters, rate, elapsed_time))
