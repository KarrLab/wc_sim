#!/bin/bash        
# 
# @author: goldbera
# run all unittests

python -m unittest discover .
# RUN coverage tests on multiple unittests
for test_file in test_*py; do
    coverage run --branch --append $test_file
done
# don't test third party libraries 
coverage html --omit "/Users/goldbera/Library/Python/2.7/lib/python/site-packages/rounding-0.03-py2.7.egg/rounding/*"