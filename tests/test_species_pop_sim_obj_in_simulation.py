'''Run a simulation with another simulation object to test SpeciesPopSimObject.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-02
:Copyright: 2016, Karr Lab
:License: MIT

A SpeciesPopSimObject manages the population of one specie, 'x'. A MockSimulationObject sends
initialization events to SpeciesPopSimObject and compares the 'x's correct population with
its simulated population.
'''
# One object (a UniversalSenderReceiverSimulationObject) sends initialization events.

import unittest

from wc_utils.util.misc import isclass_by_name
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.message_types import ALL_MESSAGE_TYPES
from wc_sim.multialgorithm.species_pop_sim_object import SpeciesPopSimObject
from wc_sim.multialgorithm.multialgorithm_errors import SpeciesPopulationError

class MockSimulationObject(SimulationObject):

    def __init__(self, name, test_case, specie_id, expected_value):
        '''Init a MockSimulationObject that can unittest a specie's population.

        Args:
            test_case (:obj:`unittest.TestCase`): reference to the TestCase that launches the simulation
        '''
        (self.test_case, self.specie_id, self.expected_value) = (test_case, specie_id, expected_value)
        super(MockSimulationObject, self).__init__(name)

    def handle_event(self, event_list):
        '''Perform a unit test on the population of self.specie_id.'''
        super(MockSimulationObject, self).handle_event(event_list)

        event_message = event_list[0]
        if isclass_by_name(event_message.event_type, message_types.GivePopulation):
            # populations is a GivePopulation_body instance
            populations = event_message.event_body
            self.test_case.assertEqual(populations[self.specie_id], self.expected_value,
                msg="At event_time {} for specie '{}': the correct population "
                            "is {} but the actual population is {}.".format(
                                event_message.event_time, self.specie_id,
                                self.expected_value, populations[self.specie_id]))
        else:
            raise SpeciesPopulationError("Error: event_message.event_type '{}' should "\
            "be covered in the if statement above".format(event_message.event_type))


class TestSpeciesPopSimObjectWithAnotherSimObject(unittest.TestCase):

    def try_update_species_pop_sim_obj(self, specie_id, init_pop, init_flux, update_message,
        msg_body, update_time, get_pop_time, expected_value):
        '''Run a simulation that tests an update of a SpeciesPopSimObject by a update_msg_type message.

        initialize simulation:
            create SpeciesPopSimObject object
            create MockSimulationObject with reference to this TestCase and expected population value
            Mock obj sends update_message for time=update_time
            Mock obj sends GetPopulation for time=get_pop_time
        run simulation:
            SpeciesPopSimObject obj processes both messages
            SpeciesPopSimObject obj sends GivePopulation
            Mock obj receives GivePopulation and checks value
        '''
        if get_pop_time<=update_time:
            raise SpeciesPopulationError('get_pop_time<=update_time')
        SimulationEngine.reset()
        species_pop_sim_obj = SpeciesPopSimObject('test_name',
            {specie_id:init_pop}, initial_fluxes={specie_id:init_flux})
        mock_obj = MockSimulationObject('mock_name', self, specie_id, expected_value)
        mock_obj.send_event(update_time, species_pop_sim_obj, update_message, event_body=msg_body)
        mock_obj.send_event(get_pop_time, species_pop_sim_obj, message_types.GetPopulation,
            event_body=message_types.GetPopulation.Body({specie_id}))
        self.assertEqual(SimulationEngine.simulate(get_pop_time+1), 3)

    def test_message_types(self):
        '''Test both discrete and continuous updates, with a range of population & flux values'''
        s_id = 's'
        update_adjustment = +5
        get_pop_time = 4
        for s_init_pop in range(3, 7, 2):
            for s_init_flux in range(-1, 2):
                for update_time in range(1, 4):

                    self.try_update_species_pop_sim_obj(s_id, s_init_pop, s_init_flux,
                        message_types.AdjustPopulationByDiscreteModel,
                        message_types.AdjustPopulationByDiscreteModel.Body({s_id:update_adjustment}),
                        update_time, get_pop_time,
                        s_init_pop + update_adjustment + get_pop_time*s_init_flux)

        '''
        Test AdjustPopulationByContinuousModel.

        Note that the expected_value does not include a term for update_time*s_init_flux. This is
        deliberately ignored by `wc_sim.multialgorithm.specie.Specie()` because it is assumed that
        an adjustment by a continuous submodel will incorporate the flux predicted by the previous
        iteration of that submodel.
        '''
        for s_init_pop in range(3, 8, 2):
            for s_init_flux in range(-1, 2):
                for update_time in range(1, 4):
                    for updated_flux in range(-1, 2):
                        self.try_update_species_pop_sim_obj(s_id, s_init_pop, s_init_flux,
                            message_types.AdjustPopulationByContinuousModel,
                            message_types.AdjustPopulationByContinuousModel.Body({s_id:
                                message_types.ContinuousChange(update_adjustment, updated_flux)}),
                            update_time, get_pop_time,
                            s_init_pop + update_adjustment +
                                (get_pop_time-update_time)*updated_flux)

MessageTypesRegistry.set_sent_message_types(MockSimulationObject, ALL_MESSAGE_TYPES)
MessageTypesRegistry.set_receiver_priorities(MockSimulationObject, ALL_MESSAGE_TYPES)
