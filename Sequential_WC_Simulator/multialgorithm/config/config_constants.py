""" Multialgorithm configuration

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-10-03
:Copyright: 2016, Karr Lab
:License: MIT
"""

import os

CONFIG_SCHEMA_FILENAME = os.path.join(os.path.dirname(__file__), 'debug_schema.cfg')
# :obj:`str`: path for configuration schema

DEFAULT_CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), 'debug_default.cfg')
# :obj:`str`: path for default configuration values

USER_CONFIG_FILENAMES = (
    'config.cfg',
    os.path.expanduser('~/.wc/config.cfg'),
)
# :obj:`list`: list of paths to search for overrides to default configuration values
