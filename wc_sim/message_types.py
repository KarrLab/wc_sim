""" A static set of message types and their content for multialgorithmic whole-cell simulations

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-06-10
:Copyright: 2016-2018, Karr Lab
:License: MIT

Simulation message types are subclasses of `SimulationMessage`.
Every simulation event message contains a typed `SimulationMessage`.

Declare
    1. For each message type which has an message, a class that represents the body

Event message types, bodies and reply message:

    AdjustPopulationByDiscreteSubmodel
        a discrete (stochastic) model increases or decreases some species copy numbers: data: dict:
        species_name -> population_change; no reply message

    AdjustPopulationByContinuousSubmodel
        a continuous model integrated by a time-step simulation increases or decreases some species
        copy numbers: data: dict: species_name -> (population_change, population_flux); no reply message

    GetPopulation
        set of species whose population is needed; data: set: species_name(s)

    GivePopulation
        response to GetPopulation; dict: species_name -> population

Define a class that stores the body of each message type. This avoids confusing the structure of a message body.
These classes should be used by all message senders and receivers.
It is enforced by checking class names against message body types.

For this sequential simulator, simulation messages are stored as a copy of or reference to sender's data structure
# TODO(Arthur): for parallel simulation, serialize and deserialize message bodies, perhaps with Pickle
"""

from collections import namedtuple

from de_sim.simulation_message import SimulationMessage


class AdjustPopulationByDiscreteSubmodel(SimulationMessage):
    """ A WC simulator message sent by a discrete time submodel to adjust species counts.

    Attributes:
        population_change (:obj:`dict` of `float`): map: species_id -> population_change;
            changes in the population of the identified species.
    """
    attributes = ["population_change"]


ContinuousChange_namedtuple = namedtuple('ContinuousChange_namedtuple', 'change, flux')


class ContinuousChange(ContinuousChange_namedtuple):
    """ A namedtuple to be used in the body of an AdjustPopulationByContinuousSubmodel message
    """

    def type_check(self):
        """ Check that the fields in ContinuousChange are numbers.

        Raises:
            ValueError: if one of the fields is non-numeric.
        """
        # https://docs.python.org/2.7/library/collections.html#collections.namedtuple documents
        # namedtuple and this approach for extending its functionality
        for f in self._fields:
            v = getattr(self, f)
            try:
                float(v)
            except:
                raise ValueError("ContinuousChange.type_check(): {} is '{}' "
                                 "which cannot be cast to a float".format(f, v))

    def __new__(cls, change, flux):
        """ Initialize a ContinuousChange.

        Raises:
            ValueError: if some fields are not numbers.
        """
        self = super().__new__(cls, change, flux)
        self.type_check()
        return self


class AdjustPopulationByContinuousSubmodel(SimulationMessage):
    """ A WC simulator message sent by a continuous submodel to adjust species counts.

    Continuous submodels model species populations as continuous variables. They're usually
    simulated by time-stepped methods. Common examples include ODEs and dynamic FBA.

    Attributes:
        population_change (:obj:`dict` of :obj:`ContinuousChange`):
            map: species_id -> ContinuousChange namedtuple; changes in the population of the
            identified species, and the predicted future flux of the species (which may be
            simply the historic flux).
    """
    attributes = ['population_change']


class GetPopulation(SimulationMessage):
    """ A WC simulator message sent by a submodel to obtain some current specie populations.

    Attributes:
        species (:obj:`set` of `str`): set of species_ids; the species whose populations are
        requested.
    """
    attributes = ['species']


class GivePopulation(SimulationMessage):
    """ A WC simulator message sent by a species pop object to report some current specie populations.

    Attributes:
        population (:obj:`dict` of `str`): species_id -> population; the populations of some species.
    """
    attributes = ['population']

# TODO(Arthur): make a pair of messages that Get and Give the population of one specie


class AggregateProperty(SimulationMessage):
    """ A WC simulator message sent to aggregate a property

    Attributes:
        property_name (:obj:`str`): the name of the requested property
    """
    attributes = ['property_name']


"""
We support two different types of get-property messages, GetCurrentProperty and GetHistoricalProperty,
with these semantics:

* GetCurrentProperty: get the value of a property at the simulation time of the event containing this
  message
* GetHistoricalProperty: get the value of a property at a time <= the simulation time of the event

Thus, a GetHistoricalProperty should be sent to a module that can provide the property's history,
at least over some time period. Handling it generates an error if the property is not available
at the requested time.
"""


class GetHistoricalProperty(SimulationMessage):
    """ A WC simulator message sent to obtain a property at a time that's not in the future

    Attributes:
        property_name (:obj:`str`): the name of the requested property
        time (`float`): the time at which the property should be measured
    """
    attributes = ['property_name', 'time']


class GetCurrentProperty(SimulationMessage):
    """ A WC simulator message sent to obtain a property at the receiver's current time

    Attributes:
        property_name (:obj:`str`): the name of the requested property
    """
    attributes = ['property_name']


class GiveProperty(SimulationMessage):
    """ A WC simulator message sent by a simulation object to report a property

    Attributes:
        property_name (:obj:`str`): the name of the reported property
        time (`float`): the time at which the property was measured
        value (:obj:`object`): the value of the property at `time`
    """
    attributes = ['property_name', 'time', 'value']


class ExecuteSsaReaction(SimulationMessage):
    """ A WC simulator message sent by a SsaSubmodel to itself to schedule an SSA reaction execution.

    Attributes:
        reaction_index (int): the index of the selected reaction in `SsaSubmodel.reactions`.
    """
    attributes = ['reaction_index']


class SsaWait(SimulationMessage):
    """ A WC simulator message sent by a SsaSubmodel to itself to temporarily suspend activity
    because no reactions are runnable.
    """


class RunFba(SimulationMessage):
    """ A WC simulator message sent by a DfbaSubmodel to itself to schedule the next FBA execution.
    """


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
