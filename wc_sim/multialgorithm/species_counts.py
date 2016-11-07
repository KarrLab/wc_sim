'''
DRAFT code for cached species counts. NOT FUNCTIONAL OR COMPLETE.
'''
from collections import namedtuple
ValueAtTime = namedtuple( 'ValueAtTime', 'value, time' )
class SpeciesCounts(object):
    """ A local species count store used by a submodel.
    
    A SpeciesCounts is a write-through cache of the count of a set of species. All of a submodel's 
    reads & writes of species counts go through a SpeciesCounts, which will synchronize the R/W operations
    with the global CellState.

    All operations occur at the current simulation time.
    
    Roadmap for SpeciesCounts:

    0. prototype: all species counts are stored in a local SpeciesCounts object
       that wrap simulation scheduled accesses to CellState
    1. locality optimization: local species counts distinguish between shared and private species
       shared species accesses all mapped into scheduled accesses to CellState
       private species accesses simply access local species counts

    Attributes:
        submodel: a reference to the submodel using this SpeciesCount
        cell_state: a reference to the CellState simulation object
        counts: dict: id -> ValueAtTime, with 
            id: string; the species' identifier
            ValueAtTime.value: float; the species' predicted copy number
            ValueAtTime.time: float; the time at which the copy number was predicted
    """
    """
    __slots__ = 'submodel cell_state counts'.split()

    def __init__( self, submodel, cell_state ):
        self.submodel = submodel
        self.cell_state = cell_state
        self.counts = {}
        # report val to CellState
    
    # TODO(Arthur): PERHAPS: map these into adjustments
    def read_val( self, id, now ):
        '''Read one value.'''
        if self.time == now:
            return self.val
        else:
            # get val from CellState
            pass
            
    def write_val( self, val, now=None ):
        '''Update the value.
        
        Called by the submodel when it a) updates the count or b) receives GivePopulation from CellState.
        '''
        if self.time == now and self.val = val:
            # nothing to update
            return
        # report val to CellState
        
    def send_to_cell_state(self):
        self.send_event( 0, receiver, event_type )
    
    """
    pass