#!/bin/bash
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


BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo """
########################
# STARTING PREPARATION #
########################
"""

#echo -e "\nCOMPILING ratio\n"
#cd "$BASEDIR/dpfinder/searcher/statistics/ratio"
#make lib
#cd -
#
#echo -e "\nCOMPILING rounder\n"
#cd "$BASEDIR/dpfinder/utils/tf/rounder"
#make lib
#cd -

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate dp-finder

export PYTHONPATH="$BASEDIR"
# ensure DP-Finder only occupies first GPU (if machine has a GPU)
export CUDA_VISIBLE_DEVICES=0

# versions
#python3 -V; python3 -c 'import tensorflow as tf; print("tensorflow",tf.__version__); import numpy; print("numpy",numpy.version.version); import scipy; print("scipy",scipy.__version__)'

echo """
########################
# FINISHED PREPARATION #
########################
"""
