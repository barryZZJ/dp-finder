#!/bin/bash
# shellcheck disable=code
# ==BEGIN LICENSE==
# 
# MIT License
# 
# Copyright (c) 2018 SRI Lab, ETH Zurich
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ==END LICENSE==

set -e # abort on errors

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASEDIR"

source ./my_prepare.sh

echo """
####################
# STARTING RUNNING #
####################
"""

# max_n_samples is 12484608, due to hardware restriction, set lower

declare -A samples=(
  ['aboveThreshold']=12946760
  ['alg1']=7018184
  ['alg2']=7018184
  ['alg3']=5851428
  ['alg4']=5851428
  ['alg5']=7018184
  ['expMech']=12484608
  ['reportNoisyMax']=12484608
  ['sum']=12484608
)

algs=('expMech' 'reportNoisyMax' 'sum' 'aboveThreshold' 'alg1' 'alg2' 'alg3' 'alg4' 'alg5')
algs=('expMech' 'reportNoisyMax' 'sum' 'aboveThreshold' 'alg1' 'alg3' 'alg4' 'alg5')

for i in {0..8} ; do
  python3 ./dpfinder/runners/tf_runner.py ${algs[i]} --opt_only --n_steps 50 --max_n_samples ${samples[${algs[i]}]} --confirming 10
done

echo """
####################
# FINISHED RUNNING #
####################
"""
