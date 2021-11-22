#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASEDIR"

for p in "logging" "runners" "utils/tf";
  do
     rm -v -r ./dpfinder/$p/logs/*
  done

