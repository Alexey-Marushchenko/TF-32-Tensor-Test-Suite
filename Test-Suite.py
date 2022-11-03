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
# Version 20221103_1016
# Target PyTorch, TensorFlow2, CUDA and CUDNN are at test files

import argparse
import functools
import subprocess
import sys
import time
import rich

print = functools.partial(print, flush=True)

# Default runs number is 13. It's wise to drop first result and use the rest
def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runs', default=13, type=int)
    parser.add_argument('-i', '--iters', default=100, type=int)
    parser.add_argument('-f', '--fp32', dest='fp32', default=False, action='store_true')
    parser.add_argument('-t', '--tf32', dest='tf32', default=False, action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true')
    parser.add_argument('--PT', dest='PT', default=False, action='store_true')
    parser.add_argument('--TF2', dest='TF2', default=False, action='store_true')
    parser.add_argument('-a', dest='all', default=False, action='store_true')

    return parser

# Main
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    from rich.console import Console
    console = Console()

    console.print(namespace)

    runs = namespace.runs
    iters = namespace.iters
    be_verbose = namespace.verbose

    tests = {}

    if namespace.PT == True:
        if namespace.fp32 == True:
            tests['PT-FP32'] = ['PyTorch FP32', 'PyTorch-Tensor-Test.py']

        if namespace.tf32 == True:
            tests['PT-TF32'] = ['PyTorch TF32', 'PyTorch-Tensor-Test.py', '-t']

    if namespace.TF2 == True:
        if namespace.fp32 == True:
            tests['TF2-FP32'] = ['TensorFlow2 FP32', 'TensorFlow2-Tensor-Test.py']

        if namespace.tf32 == True:
            tests['TF-TF32'] = ['TensorFlow2 TF32', 'TensorFlow2-Tensor-Test.py', '-t']
    
    if namespace.all == True:
        tests['PT-FP32'] = ['PyTorch FP32', 'PyTorch-Tensor-Test.py']
        tests['PT-TF32'] = ['PyTorch TF32', 'PyTorch-Tensor-Test.py', '-t']
        tests['TF2-FP32'] = ['TensorFlow2 FP32', 'TensorFlow2-Tensor-Test.py']
        tests['TF-TF32'] = ['TensorFlow2 TF32', 'TensorFlow2-Tensor-Test.py', '-t']

    tests_len = len(tests)

    if tests_len == 0:
        console.print("You need to select at least one test", style="bold red")
        sys.exit(1)

    for test_name, params in tests.items():
        tests_len -= 1
        
        if be_verbose == True:
            console.print(str(time.ctime()))
            console.print("Start of", params[0])
            console.print("-------------------------------------------------------")

            params.append('-v')

        params.append('-i ' + str(iters))

        cmd = "python3"

        for i in range(len(params)-1):
            cmd += ' ' + params[i+1]

        if be_verbose == False:
            console.print(params[0])

        for i in range(runs):
            subprocess.run(cmd.split())

        if be_verbose == True:
            console.print("End of", params[0])
            console.print(str(time.ctime()))
    
        if tests_len > 0: 
            if be_verbose == True:
                console.print("[blue]60 seconds card cooldown")

            from rich.progress import *
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                transient=True
            )

            with progress:
                task1 = progress.add_task("[blue on white]Waiting...", total=60)
                while not progress.finished:
                    time.sleep(1)
                    progress.update(task1, advance=1)

    
        if be_verbose == True:
            console.print("-------------------------------------------------------")
