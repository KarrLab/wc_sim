#!/bin/bash        
# 
# @author: goldbera
# run all unittests

python -m unittest discover .
# RUN coverage tests on multiple unittests
for test_file in test_*py; do
    coverage run --append $test_file
done
coverage html