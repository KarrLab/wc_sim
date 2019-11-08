""" A sub-model that employs a system of ordinary differential equations (ODEs) to model a set of reactions.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

import numpy as np
from scipy.constants import Avogadro
from scikits.odes import ode
import warnings
import math
from pprint import pprint

from wc_utils.util.list import det_dedupe
from wc_sim.config import core as config_core_core
from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm_errors import MultialgorithmError

config_multialgorithm = \
    config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']

'''
todos:
    Try the prerequisites in karrlab/wc_env version 0.0.45.
'''

class WcSimOdeWarning(UserWarning):
    """ `wc_sim` Ode warning """
    pass


class OdeSubmodel(DynamicSubmodel):
    """ Use a system of ordinary differential equations to predict the dynamics of chemical species in a container

    To avoid cumulative roundoff errors in event times from repeated addition, event times are
    computed by multiplying period number times period.

    Attributes:
        rate_of_change_expressions (:obj:`list`): a list of coefficient, rate law tuples for each species
        solver (:obj:`scikits.odes.ode.ode`): the Odes ode solver
        # todo: add attributes made by set_up_optimizations() and elsewhere
    """

    # register the message types sent by OdeSubmodel
    messages_sent = [message_types.RunOde]

    # register 'handle_RunOde_msg' to handle RunOde events
    event_handlers = [(message_types.RunOde, 'handle_RunOde_msg')]

    # prevent simultaneous use of multiple solver instances because of the 'OdeSubmodel.instance = self'
    # also, it's unclear whether that works; see: https://stackoverflow.com/q/34291639
    # todo: enable simultaneous use of multiple OdeSubmodel instances
    using_solver = False

    # todo: provide a dict of options that can pass through Simulate()

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
        local_species_population, time_step, testing=True, use_populations=False):
        """ Initialize an ODE submodel instance.

        Args:
            id (:obj:`str`): unique id of this dynamic ODE submodel
            dynamic_model (:obj: `DynamicModel`): the aggregate state of a simulation
            reactions (:obj:`list` of `wc_lang.Reaction`): the reactions modeled by this ODE submodel
            species (:obj:`list` of `wc_lang.Species`): the species that participate in the reactions modeled
                by this ODE submodel, with their initial concentrations
            dynamic_compartments (:obj: `dict`): `DynamicCompartment`s, keyed by id, that contain
                species which participate in reactions that this ODE submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                ODE submodel's species population
            time_step (:obj:`float`): time interval between ODE analyses
            testing (:obj:`bool`, optional): true indicates testing
            use_populations (:obj:`bool`, optional): if set, use populations instead of concentrations
        """
        # warn if reactions is empty
        if not reactions:
            warnings.warn("OdeSubmodel {}: warning, not starting because no reactions provided".format(id),
                WcSimOdeWarning)
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        if time_step <= 0:
            raise MultialgorithmError("OdeSubmodel {}: time_step must be positive, but is {}".format(
                self.id, time_step))
        if 1 < len(self.dynamic_compartments):
            raise MultialgorithmError("OdeSubmodel {}: multiple compartments not supported".format(self.id))
        # todo: migrate this count and multiply approach to computing up to DynamicSubmodel
        self.time_step = time_step
        self.time_step_count = 0
        self.testing = testing
        self.use_populations = use_populations
        self.set_up_ode_submodel()
        self.set_up_optimizations()
        self.right_hand_side_call_num = 0

    def set_up_optimizations(self):
        """For optimization, pre-compute and pre-allocate data structures"""
        # make fixed set of species ids used by this OdeSubmodel
        self.ode_species_ids_set = set(self.ode_species_ids)
        # pre-allocate dict of adjustments for LocalSpeciesPopulation
        self.adjustments = {species_id:None for species_id in self.ode_species_ids}
        # pre-allocate np arrays for concentrations and populations
        self.concentrations = np.zeros(len(self.ode_species_ids))
        self.populations = np.zeros(len(self.ode_species_ids))
        # pre-allocate dict of concentrations
        self.concentrations_dict = {species_id:0 for species_id in self.ode_species_ids}

    def get_compartment_id(self):
        # todo: discard when multiple compartments supported
        return list(self.dynamic_compartments.keys())[0]

    def get_compartment_volume(self):
        # todo: discard when multiple compartments supported
        dynamic_compartment = self.dynamic_model.dynamic_compartments[self.get_compartment_id()]
        return dynamic_compartment.volume()

    def get_concentrations(self):
        """Get current shared concentrations in numpy array"""
        specie_concentrations_dict = self.get_specie_concentrations()
        np.copyto(self.concentrations,
            [specie_concentrations_dict[id] for id in self.ode_species_ids])
        return self.concentrations

    def concentrations_to_dict(self, concentrations):
        """Convert numpy array of concentrations to dict of concentrations"""
        for id, value in zip(self.ode_species_ids, concentrations):
            self.concentrations_dict[id] = value
        return self.concentrations_dict

    def concentrations_to_populations(self, concentrations):
        """Convert numpy array of concentrations to array of populations"""
        # todo: move this to a utility
        # optimization: copy concentrations to existing self.populations &
        # modify self.populations in place with *= and out=
        np.copyto(self.populations, concentrations)
        vol_avo = self.get_compartment_volume() * Avogadro
        self.populations *= vol_avo
        return np.rint(self.populations, out=self.populations)

    def set_up_ode_submodel(self):
        """Set up an ODE submodel, including its ODE solver"""

        # HACK!: store this instance in OdeSubmodel class variable, so that right_hand_side() can use it
        OdeSubmodel.instance = self
        # disable locking temporarily
        # self.get_solver_lock()
        # find species in reactions modeled by this submodel
        ode_species = []
        for idx, rxn in enumerate(self.reactions):
            for species_coefficient in rxn.participants:
                species_id = species_coefficient.species.gen_id()
                ode_species.append(species_id)
        self.ode_species_ids = det_dedupe(ode_species)

        # this is optimal, but costs O(|self.reactions| * |rxn.participants|)
        tmp_rate_of_change_expressions = {species_id:[] for species_id in self.ode_species_ids}
        for idx, rxn in enumerate(self.reactions):
            for species_coefficient in rxn.participants:
                rate_law_id = rxn.rate_laws[0].id
                dynamic_rate_law = self.dynamic_model.dynamic_rate_laws[rate_law_id]
                species_id = species_coefficient.species.gen_id()
                tmp_rate_of_change_expressions[species_id].append((species_coefficient.coefficient,
                                                                   dynamic_rate_law))

        self.rate_of_change_expressions = []
        for species_id in self.ode_species_ids:
            self.rate_of_change_expressions.append(tmp_rate_of_change_expressions[species_id])

    def get_solver_lock(self):
        cls = self.__class__
        if not cls.using_solver:
            cls.using_solver = True
            return True
        else:
            raise MultialgorithmError("OdeSubmodel {}: cannot get_solver_lock".format(self.id))

    def release_solver_lock(self):
        cls = self.__class__
        # todo: need a mechanism for scheduling an event that calls this
        cls.using_solver = False
        return True

    def set_up_ode_solver(self):
        """Set up the `scikits.odes` ODE solver"""
        # todo: optimization: methods in DynamicSubmodel and LocalSpeciesPopulation to put
        # populations/concentrations directly into existing np arrays
        if self.use_populations:
            specie_populations_dict = self.get_specie_counts(round=False)
            self.populations = np.asarray([specie_populations_dict[id] for id in self.ode_species_ids])
            # use CVODE from LLNL's SUNDIALS (https://computation.llnl.gov/projects/sundials)
            self.solver = ode('cvode', self.right_hand_side, old_api=False)
            solver_return = self.solver.init_step(self.time, self.populations)
        else:
            specie_concentrations_dict = self.get_specie_concentrations()
            self.concentrations = np.asarray([specie_concentrations_dict[id] for id in self.ode_species_ids])
            # use CVODE
            self.solver = ode('cvode', self.right_hand_side, old_api=False)
            solver_return = self.solver.init_step(self.time, self.concentrations)
        if not solver_return.flag:
            raise MultialgorithmError("OdeSubmodel {}: solver.init_step() failed: '{}'".format(self.id,
                solver_return.message)) # pragma: no cover
        return solver_return

    @staticmethod
    def right_hand_side(time, species_state, species_state_change_rates):
        """Evaluate concentration change rates for all species; called by ODE solver

        Args:
            time (:obj:`float`): simulation time
            species_state (:obj:`numpy.ndarray`): species_state of species at time `time`, in the
                same order as `self.species`
            species_state_change_rates (:obj:`numpy.ndarray`): the rate of change of species_state at
                time `time`; written by this method

        Returns:
            :obj:`int`: return 0 to indicate success, 1 to indicate failure;
                see http://bmcage.github.io/odes/version-v2.3.2/ode.html#scikits.odes.ode.ode;
                but the important side effects are the values in `species_state_change_rates`
        """
        # this is called by a c code wrapper that's used by the CVODE ODE solver
        try:
            # obtain hacked instance reference
            self = OdeSubmodel.instance
            self.right_hand_side_call_num += 1

            # if self.use_populations, then use populations instead of species_state
            if self.use_populations:
                pass
            else:
                # combine concentrations from solver with state values
                # todo: carefully think about when to grab populations/concentrations, & schedule timesteps
                species_concentrations = self.get_specie_concentrations()
                species_state_dict = self.concentrations_to_dict(species_state)
                # todo: only need concentrations for species used in this submodel's rate laws
                for id, val in species_concentrations.items():
                    if id not in species_state_dict:
                        species_state_dict[id] = val

                # todo: optimize by using self.calc_reaction_rates()
                # for each specie in `species_state` sum evaluations of rate laws in self.rate_of_change_expressions
                for idx, conc in enumerate(np.nditer(species_state)):
                    specie_rxn_rates = []
                    for coeff, dyn_rate_law in self.rate_of_change_expressions[idx]:
                        specie_rxn_rates.append(coeff * dyn_rate_law.eval(
                            self.time,
                            parameter_values=self.get_parameter_values(),
                            species_concentrations=species_state_dict))
                    species_state_change_rates[idx] = sum(specie_rxn_rates)

            if False and math.log2(self.right_hand_side_call_num).is_integer():
                print("@{:.4E}: rhs #: {}".format(time, self.right_hand_side_call_num))
                print('species_state_dict')
                pprint(species_state_dict)
                print('species_state_change_rates')
                pprint(species_state_change_rates)

            return 0
        except Exception as e:
            if self.testing:
                raise MultialgorithmError("OdeSubmodel {}: solver.right_hand_side() failed: '{}'".format(
                    self.id, e))
            return 1

    def run_ode_solver(self):
        """Run the ODE solver for one time step and save its results"""
        ### run the ODE solver
        # re-initialize the solver to include changes in concentrations by other submodels
        # must be done each time solver.step() is called
        self.set_up_ode_solver()
        # minimize round-off error for time by counting steps and multiplying time step * num steps
        end_time = self.time_step_count * self.time_step
        solution = self.solver.step(end_time)
        if solution.flag:
            raise MultialgorithmError("OdeSubmodel {}: solver step() error: '{}' for time step [{}, {})".format(
                self.id, solution.message, self.time, end_time))

        ### store results in local_species_population
        '''
        approach
            pre-compute mean rate of population change for the next time step
                init_pops = initial population at start of this ODE analysis
                    solution.values.y is an np array w shape 1xnum(species)
                curr_pops = solution.values.y converted to pops
                pops_change = curr_pops - init_pops

            rate = pops_change/self.time_step
            map all to dict (pre-allocated)
        '''
        # if use_populations, then use populations instead of concentrations
        if self.use_populations:
            pass
        else:
            # todo: optimization: optimize LocalSpeciesPopulation to provide an array
            init_pops_dict = self.local_species_population.read(self.time, self.ode_species_ids_set)
            # todo: optimization: after optimizing LocalSpeciesPopulation, optimize these
            init_pops_list = [init_pops_dict[species_id] for species_id in self.ode_species_ids]
            init_pops_array = np.array(init_pops_list)

            # convert concentrations to populations
            curr_pops = self.concentrations_to_populations(solution.values.y)
            pops_change = curr_pops - init_pops_array

            rate = pops_change / self.time_step
            for idx, species_id in enumerate(self.ode_species_ids):
                self.adjustments[species_id] = rate[idx]
            # todo: optimization: optimize LocalSpeciesPopulation to accept arrays
            self.local_species_population.adjust_continuously(self.time, self.adjustments)

    # schedule and handle events
    def send_initial_events(self):
        """Send this ODE submodel's initial event"""
        self.schedule_next_ode_analysis()

    def increment_time_step_count(self):
        self.time_step_count += 1

    def schedule_next_ode_analysis(self):
        """Schedule the next analysis by this ODE submodel"""
        self.send_event_absolute(self.time_step_count * self.time_step, self, message_types.RunOde())
        self.time_step_count += 1

    def handle_RunOde_msg(self, event):
        """Handle an event containing a RunOde message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_ode_solver()
        self.schedule_next_ode_analysis()
