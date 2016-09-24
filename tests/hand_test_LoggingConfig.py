#!/usr/bin/env python

import logging
from Sequential_WC_Simulator.core.logging_config import setup_logger_old

# TODO(Arthur): automate this test
for stdout in [True, False]:
    logger_name = 'on_stdout_{}'.format( stdout )
    setup_logger_old(logger_name, level=logging.DEBUG, to_stdout=stdout)
    my_logger = logging.getLogger(logger_name)
    my_logger.debug('On stdout == {}'.format( stdout ) )
