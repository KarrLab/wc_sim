'''An abc that defines the interface between a submodel and its species population stores.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-04
:Copyright: 2016, Karr Lab
:License: MIT
'''

import abc, six

@six.add_metaclass(abc.ABCMeta)
class AccessSpeciesPopulationInterface():
    '''An abstract base class defining the interface between a submodel and its species population stores.
    
    A submodel in a parallel WC simulation will interact with multiple components that store the
    population of species it accesses. All these stores should implement this interface which
    defines read and write operations on the species in a store.
    '''
    
    @abc.abstractmethod
    def read_one(self, time, specie_id):
        '''Read the predicted population of a specie at a particular time.'''
        pass

    @abc.abstractmethod
    def read( self, time, species ):
        '''Read the predicted population of a list of species at a particular time.'''
        pass

    @abc.abstractmethod
    def adjust_discretely( self, time, adjustments ):
        '''A discrete model adjusts the population of a set of species at a particular time.'''
        pass

    @abc.abstractmethod
    def adjust_continuously( self, time, adjustments ):
        '''A continuous model adjusts the population of a set of species at a particular time.'''
        pass
