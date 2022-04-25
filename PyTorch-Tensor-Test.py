#!/usr/bin/python3
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
# Version 20220425_1531
# Target torch version 1.11.0
# Target CUDA version 11.6.2
# Target CUDNN version 8.4.0

import argparse
import functools
import os
import sys
import time
import torch

dtype = torch.float32

print = functools.partial(print, flush=True)


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-i', '--iters', nargs='?', default=100, type=int)
    parser.add_argument ('-s', '--size', nargs='?', default=8192, type=int)
    parser.add_argument ('-t', '--tensor', dest='tensor', default=False, action='store_true')
    parser.add_argument ('-v', '--verbose', dest='verbose', default=False, action='store_true')
 
    return parser

# Main
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()

    iters = namespace.iters
    n = namespace.size
    use_tensor_cores = namespace.tensor
    be_verbose = namespace.verbose

    cuda = torch.device('cuda:0')
    g_cuda=torch.Generator(cuda)
    g_cuda.seed()

    in_matrix = torch.rand(n, n, generator=g_cuda, dtype=dtype, device=cuda)
    ret_matrix = torch.rand(n, n, generator=g_cuda, dtype=dtype, device=cuda)

    torch.backends.cuda.matmul.allow_tf32 = use_tensor_cores
    torch.backends.cudnn.allow_tf32 = use_tensor_cores

    if be_verbose == True:
        print("\nIs TF32 matmul enabled: " + str(torch.backends.cuda.matmul.allow_tf32))
        print("Is TF32 cudnn enabled: " + str(torch.backends.cudnn.allow_tf32))

    # Warmup with at least 1200 iterations
    if be_verbose == True:
        print('\nKeep calm while warming up the card...\n')

    for i in range(1200):
        ret_matrix = in_matrix @ ret_matrix

    start_time = time.time()
    iter_time = start_time
 
    for i in range(1, iters+1):
        ret_matrix = in_matrix @ ret_matrix

        if be_verbose == True and (i/iters*100).is_integer():
            tran_time = time.time()
            print(str(time.ctime()) + ' | Iter num %d of %d | %.2f ' % (i, iters, i/iters*100) + r"%" + ' done | Est. time remain is %.2f sec.' % ((tran_time - iter_time)*(1-i/iters)*100))
            iter_time=tran_time

    end_time = time.time()

    ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
    elapsed_time = (end_time - start_time)
    rate = iters*ops/elapsed_time/10**9

    if be_verbose == True:
        print(str(time.ctime()))
        print("\nWas TF32 matmul enabled: " + str(torch.backends.cuda.matmul.allow_tf32))
        print("Was TF32 cudnn enabled: " + str(torch.backends.cudnn.allow_tf32))

    print('\n%d x %d matmul took: %.2f sec per iter at rate %.2f G ops/sec. Total time is: %.2f sec' % (n, n, elapsed_time/iters, rate, elapsed_time))
