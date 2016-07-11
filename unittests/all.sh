#!/bin/bash        
# 
# @author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
# run all unittests

function usage {
    if [ -n "$1" ]; then echo $1 >&2; fi
cat <<EOF >&2
usage: $(basename $0) 
  run all unittests for Sequential_WC_Simulator
  -c: perform coverage tests as well
  -h: produce this message 
EOF
    exit 1
}

OPT_COMBINE=''
while getopts ":ch" opt; do
    case $opt in
    c)
        OPT_COVERAGE=1
        ;;
	h)
	    usage
	    ;;
	\?)
	    usage "unknown argument: -$OPTARG"
	    ;;
	:)
	    usage "value required for argument: -$OPTARG"
	    ;;
    esac
done

shift $(($OPTIND - 1))

python -m unittest discover .

if [[ -n "$OPT_COVERAGE" ]]
then
    # run coverage tests on all unittests
    coverage erase
    # TODO: have coverage test print a summary
    for test_file in test_*py; do
        coverage run --branch --append $test_file
    done
    # don't test third party libraries 
    # TODO: make this portable
    coverage html --omit "/Users/goldbera/Library/Python/2.7/lib/python/site-packages/rounding-0.03-py2.7.egg/rounding/*"
fi

