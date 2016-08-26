#!/bin/bash        
# 
# @author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
# coverage test MultiAlgorithm.py

cd "$HOME/gitOnMyLaptop/Sequential_WC_Simulator"
coverage erase
coverage run --branch  --omit="*__init__.py"  ./multialgorithm/MultiAlgorithm.py \
"$HOME/gitOnMyLaptop/WcModelingTutorial/Exercise 3 -- Multi-algorithm simulation/Solution/Model.xlsx" 3000 --debug
coverage html --omit "$HOME/Library/Python/2.7/lib/python/site-packages/cobra/*"
