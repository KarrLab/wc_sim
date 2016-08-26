#!/usr/bin/env python3

import logging
from Sequential_WC_Simulator.core.LoggingConfig import setup_logger

# TODO(Arthur): automate this test
for stdout in [True, False]:
    logger_name = 'on_stdout_{}'.format( stdout )
    setup_logger(logger_name, level=logging.DEBUG, to_stdout=stdout)
    my_logger = logging.getLogger(logger_name)
    my_logger.debug('On stdout == {}'.format( stdout ) )
