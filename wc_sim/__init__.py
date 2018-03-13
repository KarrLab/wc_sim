import pkg_resources

with open(pkg_resources.resource_filename('wc_sim', 'VERSION'), 'r') as file:
    __version__ = file.read().strip()
# :obj:`str`: version

# API
from . import core
from . import log
from . import multialgorithm
from . import sim_config
from . import sim_metadata
