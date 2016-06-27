#!/bin/bash        
# 
# @author: goldbera
# run all unittests

python -m unittest discover .
# TODO: figure out why 'coverage run test_*py' does not work
for test_file in test_*py; do
    coverage run $test_file
done
