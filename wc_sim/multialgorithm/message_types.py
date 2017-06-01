'''A static set of message types and their content for multialgorithmic whole-cell simulations.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-06-10
:Copyright: 2016, Karr Lab
:License: MIT

Simulation message types are subclasses of `SimulationMessage`, defined by `SimulationMsgUtils.create()`.
Every simulation event message contains a typed `SimulationMessage`.

For this sequential simulator, simulation messages are stores as a copy of or reference to sender's data structure
# TODO(Arthur): for parallel simulation, serialize and deserialize message bodies, perhaps with Pickle
'''

from wc_sim.core.simulation_message import SimulationMsgUtils
from collections import namedtuple

AdjustPopulationByDiscreteSubmodel = SimulationMsgUtils.create('AdjustPopulationByDiscreteSubmodel',
    '''A WC simulator message sent by a discrete time submodel to adjust species counts.

        Attributes:
            population_change (:obj:`dict` of `float`): map: species_id -> population_change;
            changes in the population of the identified species.''',
    ["population_change"])

ContinuousChange_namedtuple = namedtuple('ContinuousChange_namedtuple', 'change, flux')
class ContinuousChange(ContinuousChange_namedtuple):
    '''A namedtuple to be used in the body of an AdjustPopulationByContinuousSubmodel message.'''

    def type_check(self):
        '''Check that the fields in ContinuousChange are numbers.

        Raises:
            ValueError: if one of the fields is non-numeric.
        '''
        # https://docs.python.org/2.7/library/collections.html#collections.namedtuple documents
        # namedtuple and this approach for extending its functionality
        for f in self._fields:
            v = getattr(self,f)
            if not (isinstance(v, int) or isinstance(v, float)):
                raise ValueError("ContinuousChange.type_check(): {} is '{}' "
                    "which is not an int or float".format(f, v))

    def __new__(cls, change, flux):
        '''Initialize a ContinuousChange.

        Raises:
            ValueError: if some fields are not numbers.
        '''
        self = super(ContinuousChange, cls).__new__(cls, change, flux)
        self.type_check()
        return self

AdjustPopulationByContinuousSubmodel = SimulationMsgUtils.create('AdjustPopulationByContinuousSubmodel',
    '''A WC simulator message sent by a continuous submodel to adjust species counts.

    Continuous submodels model species populations as continuous variables. They're usually
    simulated by time-stepped methods. Common examples include ODEs and dynamic FBA.

    Attributes:
        population_change (:obj:`dict` of :obj:`ContinuousChange`):
            map: species_id -> ContinuousChange namedtuple; changes in the population of the
            identified species, and the predicted future flux of the species (which may be
            simply the historic flux).''',
    ['population_change'])

GetPopulation = SimulationMsgUtils.create('GetPopulation',
    '''A WC simulator message sent by a submodel to obtain some current specie populations.

    Attributes:
        species (:obj:`set` of `str`): set of species_ids; the species whose populations are
        requested.
    ''',
    ['species'])

GivePopulation = SimulationMsgUtils.create('GivePopulation',
    '''A WC simulator message sent by a species pop object to report some current specie populations.

        Attributes:
            population (:obj:`dict` of `str`): species_id -> population; the populations of some species.
    ''',
    ['population'])

# TODO(Arthur): make a pair of messages that Get and Give the population of one specie

AggregateProperty = SimulationMsgUtils.create('AggregateProperty',
    '''A WC simulator message sent to aggregate a property

        Attributes:
            property_name (:obj:`str`): the name of the requested property
    ''',
    ['property_name'])

'''
We support two different types of get-property messages, GetCurrentProperty and GetHistoricalProperty,
with these semantics:
* GetCurrentProperty: get the value of a property at the simulation time of the event containing this
    message
* GetHistoricalProperty: get the value of a property at a time <= the simulation time of the event
Thus, a GetHistoricalProperty should be sent to a module that can provide the property's history,
at least over some time period. Handling it generates an error if the property is not available
at the requested time.
'''

GetHistoricalProperty = SimulationMsgUtils.create('GetHistoricalProperty',
    '''A WC simulator message sent to obtain a property at a time that's not in the future

        Attributes:
            property_name (:obj:`str`): the name of the requested property
            time (`float`): the time at which the property should be measured
    ''',
    ['property_name', 'time'])

GetCurrentProperty = SimulationMsgUtils.create('GetCurrentProperty',
    '''A WC simulator message sent to obtain a property at the receiver's current time

        Attributes:
            property_name (:obj:`str`): the name of the requested property
    ''',
    ['property_name'])

GiveProperty = SimulationMsgUtils.create('GiveProperty',
    '''A WC simulator message sent by a simulation object to report a property

        Attributes:
            property_name (:obj:`str`): the name of the reported property
            time (`float`): the time at which the property was measured
            value (:obj:`object`): the value of the property at `time`
    ''',
    ['property_name', 'time', 'value'])

ExecuteSsaReaction = SimulationMsgUtils.create('ExecuteSsaReaction',
    '''A WC simulator message sent by a SsaSubmodel to itself to schedule an SSA reaction execution.

        Attributes:
            reaction_index (int): the index of the selected reaction in `SsaSubmodel.reactions`.
    ''',
    ['reaction_index'])

SsaWait = SimulationMsgUtils.create('SsaWait',
    '''A WC simulator message sent by a SsaSubmodel to itself to temporarily suspend activity
        because no reactions are runnable.
    ''')

RunFba = SimulationMsgUtils.create('RunFba',
    '''A WC simulator message sent by a DfbaSubmodel to itself to schedule the next FBA execution.
    ''')

ALL_MESSAGE_TYPES = [
    AdjustPopulationByDiscreteSubmodel,    # A discrete model changes the population.
    AdjustPopulationByContinuousSubmodel,  # A continuous model changes the population.
    GetPopulation,                      # A submodel requests populations from a
                                        # species population simulation object.
    GivePopulation,                     # A species population simulation object provides
                                        # populations to a submodel.
    ExecuteSsaReaction,                 # An SSA submodel schedules its next reaction.
    SsaWait,                            # An SSA submodel with 0 total propensity schedules
                                        # a future effort to schedule a reaction.
    RunFba,                             # An FBA submodel schedules its next computation.
    AggregateProperty,                  # Aggregate a property
    GetCurrentProperty,                 # get the value of a property at the current simulation time
    GetHistoricalProperty,              # get a property's value at a time <= the current simulation time
    GiveProperty]                       # Provide a property to a requestor
