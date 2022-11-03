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
# Version 20221103_1017
# Target Tensorflow version 2.10.0
# Target CUDA version 11.8.0
# Target CUDNN version 8.6.0

import argparse
import distutils
import functools
import os
import sys
import time
import rich

# Tensorflow spams a lot to stderr
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

dtype = tf.float32

print = functools.partial(print, flush=True)

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
            
    device_spec = tf.DeviceSpec.from_string("/GPU:0")
    with tf.device(device_spec.to_string()):
        in_matrix = tf.random.uniform((n, n), minval=-1, maxval=1, dtype=dtype)
        ret_matrix = tf.random.uniform((n, n), minval=-1, maxval=1, dtype=dtype)

    style_color = "bold red"
    if use_tensor_cores == True:
        style_color = "bold green"

    from rich.console import Console
    console = Console()

    if be_verbose == True:
        console.print("Is TF32 enabled:", str(tf.config.experimental.tensor_float_32_execution_enabled()), style=style_color)

    # Warmup with at least 1200 iterations
    if be_verbose == True:
        tasks = [f"task {n}" for n in range(1200)]
        with console.status("[blue on white]Keep calm while warming up the card...", spinner="circleQuarters") as status:
            while tasks:
                task = tasks.pop(0)
                ret_matrix = mul(in_matrix, ret_matrix)
    else:
        for i in range(1200):
            ret_matrix = mul(in_matrix, ret_matrix)

    if be_verbose == True:
        console.print("Test start time:", str(time.ctime()))
    
    start_time = time.time()
    
    from rich.progress import *
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True
    )

    with progress:
        task1 = progress.add_task("[blue]Working...", total=iters)
        while not progress.finished:
            ret_matrix = mul(in_matrix, ret_matrix)
            progress.update(task1, advance=1)
    
    end_time = time.time()
    
    ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
    elapsed_time = (end_time - start_time)
    rate = iters*ops/elapsed_time/10**9

    if be_verbose == True:
        console.print("Test end time:", str(time.ctime()))

    console.print('%d x %d matmul took: %.4f sec per iter at rate %.2f G ops/sec. Total time is: %.4f sec' % (n, n, elapsed_time/iters, rate, elapsed_time))
