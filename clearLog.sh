#!/bin/bash
set -e # abort on errors

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASEDIR"

paths=("logging" "runners" "utils/tf")

for p in $paths;
  do
     rm -v -r ./dpfinder/$p/logs/*
  done

