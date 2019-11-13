""" A submodel that uses a system of ordinary differential equations (ODEs) to model a set of reactions.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

from pprint import pprint
from scikits.odes import ode
from scipy.constants import Avogadro
import math
import numpy as np
import warnings

from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_core
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_utils.util.list import det_dedupe

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']

# todo: make a config variable
NUM_INTERNAL_STEPS_PER_TIME_STEP = 50


class WcSimOdeWarning(UserWarning):
    """ `wc_sim` Ode warning """
    pass


class OdeSubmodel(DynamicSubmodel):
    """ Use a system of ordinary differential equations to predict the dynamics of chemical species in a container

    Attributes:
        time_step (:obj:`float`): the time between ODE solutions
        solver (:obj:`scikits.odes.ode.ode`): the Odes ode solver
        ode_species_ids (:obj:`list`): ids of the species used by this ODE solver
        ode_species_ids_set (:obj:`set`): ids of the species used by this ODE solver
        num_species (:obj:`int`): number of species used by this ODE solver
        populations (:obj:`numpy.ndarray`): pre-allocated numpy arrays for storing species populations
        rate_of_change_expressions (:obj:`list` of :obj:`list` of :obj:`tuple`): for each species,
            a list of its (coefficient, rate law) pairs
        adjustments (:obj:`dict`): pre-allocated adjustments for passing changes to LocalSpeciesPopulation
        num_species (:obj:`int`): number of species in `ode_species_ids`
        num_right_hand_side_calls (:obj:`int`): number of calls to `right_hand_side` in a call to
            the solver, `self.solver.solve()`
        history_num_right_hand_side_calls (:obj:`list` of :obj:`tuple`): history of number of calls
            to `right_hand_side`
    """

    # register the message types sent by OdeSubmodel
    messages_sent = [message_types.RunOde]

    # register 'handle_RunOde_msg' to handle RunOde events
    event_handlers = [(message_types.RunOde, 'handle_RunOde_msg')]

    # prevent simultaneous use of multiple solver instances because of the 'OdeSubmodel.instance = self'
    # also, it's unclear whether that works; see: https://stackoverflow.com/q/34291639
    # todo: enable simultaneous use of multiple OdeSubmodel instances
    using_solver = False

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
        local_species_population, time_step, testing=True):
        """ Initialize an ODE submodel instance

        Args:
            id (:obj:`str`): unique id of this dynamic ODE submodel
            dynamic_model (:obj: `DynamicModel`): the aggregate state of a simulation
            reactions (:obj:`list` of `wc_lang.Reaction`): the reactions modeled by this ODE submodel
            species (:obj:`list` of `wc_lang.Species`): the species that participate in the reactions
                modeled by this ODE submodel
            dynamic_compartments (:obj: `dict`): `DynamicCompartment`s, keyed by id, that contain
                species which participate in reactions that this ODE submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                ODE submodel's species population
            time_step (:obj:`float`): initial time interval between ODE analyses
            testing (:obj:`bool`, optional): true indicates testing
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        if time_step <= 0:
            raise MultialgorithmError(f"OdeSubmodel {self.id}: time_step must be positive, but is {time_step}")
        self.time_step = time_step
        self.testing = testing
        self.set_up_ode_submodel()
        self.set_up_optimizations()
        self.solver = self.create_ode_solver()
        self.num_right_hand_side_calls = 0
        self.history_num_right_hand_side_calls = []

    def set_up_ode_submodel(self):
        """ Set up an ODE submodel, including its ODE solver """

        # store this instance in OdeSubmodel class variable, so that right_hand_side() can use it
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
        tmp_coeffs_and_rate_laws = {species_id:[] for species_id in self.ode_species_ids}
        for idx, rxn in enumerate(self.reactions):
            rate_law_id = rxn.rate_laws[0].id
            dynamic_rate_law = self.dynamic_model.dynamic_rate_laws[rate_law_id]
            for species_coefficient in rxn.participants:
                species_id = species_coefficient.species.gen_id()
                tmp_coeffs_and_rate_laws[species_id].append((species_coefficient.coefficient,
                                                             dynamic_rate_law))

        self.rate_of_change_expressions = []
        for species_id in self.ode_species_ids:
            self.rate_of_change_expressions.append(tmp_coeffs_and_rate_laws[species_id])

    def set_up_optimizations(self):
        """ For optimization, pre-compute and pre-allocate some data structures """
        # make fixed set of species ids used by this OdeSubmodel
        self.ode_species_ids_set = set(self.ode_species_ids)
        # pre-allocate dict of adjustments used to pass changes to LocalSpeciesPopulation
        self.adjustments = {species_id:None for species_id in self.ode_species_ids}
        # pre-allocate numpy arrays for populations
        self.num_species = len(self.ode_species_ids)
        self.populations = np.zeros(self.num_species)

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

    def create_ode_solver(self):
        """ Create a `scikits.odes` ODE solver that uses CVODE

        Returns:
            :obj:`scikits.odes.ode`: an ODE solver instance

        Raises:
            :obj:`MultialgorithmError`: if the ODE solver cannot be created
        """
        # use CVODE from LLNL's SUNDIALS project (https://computing.llnl.gov/projects/sundials)
        CVODE_SOLVER = 'cvode'
        solver = ode(CVODE_SOLVER, self.right_hand_side, old_api=False)
        if not isinstance(solver, ode):    # pragma: no cover
            raise MultialgorithmError(f"OdeSubmodel {self.id}: scikits.odes.ode() failed")
        return solver

    @staticmethod
    def right_hand_side(time, species_populations, population_change_rates):
        """ Evaluate population change rates for species modeled by ODE; called by ODE solver

        Args:
            time (:obj:`float`): simulation time
            species_populations (:obj:`numpy.ndarray`): populations of all species at time `time`,
                listed in the same order as `self.ode_species_ids`
            population_change_rates (:obj:`numpy.ndarray`): the rate of change of species_populations
                at time `time`; written by this method

        Returns:
            :obj:`int`: return 0 to indicate success, or 1 to indicate failure;
                but the important side effects are the values in `population_change_rates`
        """
        # this is called by the CVODE ODE solver
        # todo: need to update a LocalSpeciesPopulation with the values in species_populations
        # or request VERY short time steps with very few calls to right_hand_side
        # alternatives:
        # exact ODE: have LocalSpeciesPopulation store a history back to the latest ODE execution
        # and use the history for other submodels
        # approximate ODE: create a cache LocalSpeciesPopulation for this OdeSubmodel and its rate laws
        # update it with changes to the population passed by the solver in species_populations
        # todo: make `self.time_step` dynamic, with longer time steps when the ODE solver uses larger internal steps
        # and shorter ones when it uses shorter internal steps; tune `self.time_step` so the number of
        # internal steps per time step approximates NUM_INTERNAL_STEPS_PER_TIME_STEP
        try:
            # obtain the OdeSubmodel instance
            self = OdeSubmodel.instance
            self.num_right_hand_side_calls += 1

            # todo: perhaps: optimization: calculate each rate law only once
            # possible approach
            # 1. make constant, sparse stoichiometric matrix Srs of the reactions integrated by ODE submodel
            # 2. calculate all rate laws in Ar
            # 3. get rate(species) = sum over species s(Ar * Srs)
            for idx in range(self.num_species):
                species_rxn_rate = 0.0
                for coeff, dyn_rate_law in self.rate_of_change_expressions[idx]:
                    species_rxn_rate += coeff * dyn_rate_law.eval(self.time)
                population_change_rates[idx] = species_rxn_rate

            return 0

        except Exception as e:
            # todo: this exception should always be raised,
            if self.testing:
                raise MultialgorithmError("OdeSubmodel {}: solver.right_hand_side() failed: '{}'".format(
                    self.id, e))

            return 1

    def create_local_species_populations(self):
        """ Create a temporary :obj:`LocalSpeciesPopulation`

        Returns:
            :obj:`scikits.odes.ode`: an ODE solver instance

        Raises:
            :obj:`MultialgorithmError`: if the ODE solver cannot be created
        """
        # todo: remove or implement
        pass

    def current_species_populations(self):
        """ Obtain the current populations of species modeled by this ODE

        The current populations are written into `self.populations`.
        """
        # todo: optimization: have LocalSpeciesPopulation provide an array and use it here
        pops_dict = self.local_species_population.read(self.time, self.ode_species_ids_set)
        for idx, species_id in enumerate(self.ode_species_ids):
            self.populations[idx] = pops_dict[species_id]

    def run_ode_solver(self):
        """ Run the ODE solver for one WC simulator time step and save the species populations changes """

        ### run the ODE solver ###
        end_time = self.time + self.time_step
        # advance one step
        solution_times = [self.time, end_time]
        self.num_right_hand_side_calls = 0
        self.current_species_populations()
        solution = self.solver.solve(solution_times, self.populations)
        if solution.flag:   # pragma: no cover
            raise MultialgorithmError("OdeSubmodel {}: solver step() error: '{}' for time step [{}, {})".format(
                self.id, solution.message, self.time, end_time))

        ### store results in local_species_population ###
        population_changes = solution.values.y[1] - self.populations
        print('population_changes:\n',population_changes)
        population_change_rates = population_changes / self.time_step
        print('population_change_rates:\n',population_change_rates)
        print('self.ode_species_ids', self.ode_species_ids)
        for idx, species_id in enumerate(self.ode_species_ids):
            self.adjustments[species_id] = (0, population_change_rates[idx])
        print('self.adjustments')
        pprint(self.adjustments)
        # todo: optimization: have LocalSpeciesPopulation.adjust_continuously() take an array
        self.local_species_population.adjust_continuously(self.time, self.adjustments)
        self.history_num_right_hand_side_calls.append(self.num_right_hand_side_calls)
        print(f"Num rhs calls\t{self.num_right_hand_side_calls}")

    ### schedule and handle DES events ###
    def send_initial_events(self):
        """ Send this ODE submodel's initial event """
        self.schedule_next_ode_analysis()

    def schedule_next_ode_analysis(self):
        """ Schedule the next analysis by this ODE submodel """
        self.send_event(self.time_step, self, message_types.RunOde())

    def handle_RunOde_msg(self, event):
        """ Handle an event containing a RunOde message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_ode_solver()
        self.schedule_next_ode_analysis()
