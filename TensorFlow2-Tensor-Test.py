#!/usr/bin/python3
#
# Copyright (C) 2022 Alexey Marushchenko <alexey@marush.com>
#
# Based on https://github.com/yaroslavvb/stuff/blob/master/matmul_benchmark.py
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
# Version 20220425_1402
# Target Tensorflow version 2.8.0
# Target CUDA version 11.6.2
# Target CUDNN version 8.4.0

import argparse
import distutils
import functools
import os
import sys
import tensorflow as tf
import time

dtype = tf.float32

print = functools.partial(print, flush=True)

# Tensorflow spams a lot to stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Debug to check where ops run
#tf.debugging.set_log_device_placement(True)


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-i', '--iters', nargs='?', default=100, type=int)
    parser.add_argument ('-s', '--size', nargs='?', default=8192, type=int)
    parser.add_argument ('-t', '--tensor', dest='tensor', default=False, action='store_true')
    parser.add_argument ('-v', '--verbose', dest='verbose', default=False, action='store_true')

    return parser


def mul(matrix1, matrix2):
    device_spec = tf.DeviceSpec.from_string("/GPU:0")
    with tf.device(device_spec.to_string()):
        return tf.linalg.matmul(matrix1, matrix2)

# Main
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()
    
    iters = namespace.iters
    n = namespace.size
    use_tensor_cores = namespace.tensor
    be_verbose = namespace.verbose

    # Avoid any optimizations
    tf.config.optimizer.set_experimental_options({'disable_model_pruning': True, 'disable_meta_optimizer': True})

    tf.config.experimental.enable_tensor_float_32_execution(use_tensor_cores)

    if be_verbose == True:
        print("\nIs TF32 enabled: " + str(tf.config.experimental.tensor_float_32_execution_enabled()))

    device_spec = tf.DeviceSpec.from_string("/GPU:0")
    with tf.device(device_spec.to_string()):
        in_matrix = tf.random.uniform((n, n), minval=-1, maxval=1, dtype=dtype)
        ret_matrix = tf.random.uniform((n, n), minval=-1, maxval=1, dtype=dtype)

    # Warmup with at least 1200 iterations
    if be_verbose == True:
        print('\nKeep calm while warming up the card...\n')

    for i in range(1200):
        ret_matrix = mul(in_matrix, ret_matrix)

    start_time = time.time()
    iter_time = start_time
 
    for i in range(1, iters+1):
        ret_matrix = mul(in_matrix, ret_matrix)

        if be_verbose == True and (i/iters*100).is_integer():
            tran_time = time.time()
            print(str(time.ctime()) + ' | Iter num %d of %d | %.2f ' % (i, iters, i/iters*100) + r"%" + ' done | Est. time remain is %.2f sec.' % ((tran_time - iter_time)*(1-i/iters)*100), flush=True)
            iter_time=tran_time

    end_time = time.time()

    ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
    elapsed_time = (end_time - start_time)
    rate = iters*ops/elapsed_time/10**9

    if be_verbose == True:
        print(str(time.ctime()))
        print("\nWas TF32 enabled: " + str(tf.config.experimental.tensor_float_32_execution_enabled()))

    print('\n%d x %d matmul took: %.2f sec per iter at rate %.2f G ops/sec. Total time is: %.2f sec' % (n, n, elapsed_time/iters, rate, elapsed_time))
