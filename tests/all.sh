#!/bin/bash        
# 
# @author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
# run all unittests

# TODO(Arthur): IMPORTANT: almost all unittests should be in multialgorithm, which they test
# TODO(Arthur): IMPORTANT: make sure that each unit test succeeds on its own

function usage {
    if [ -n "$1" ]; then echo $1 >&2; fi
cat <<EOF >&2
usage: $(basename $0) 
  run all unittests for wc_sim
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -n "$OPT_COVERAGE" ]]
then
    nosetests $SCRIPT_DIR --with-coverage --cover-package=wc_sim
    coverage html
else
    nosetests $SCRIPT_DIR
fi
