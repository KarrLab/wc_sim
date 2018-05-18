'''Define distributed properties of multi-algorithmic DES simulations of whole-cell models

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-05-30
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

from wc_sim.multialgorithm.aggregate_distributed_props import DistributedPropertyFactory

# todo: test this with species population objects
obtain_molecular_mass =  DistributedPropertyFactory.make_distributed_property(
    'MASS',
    'obtain_mass',
    100,    # seconds in period
    'sum')
