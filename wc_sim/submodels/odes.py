""" A submodel that uses a system of ordinary differential equations (ODEs) to model a set of reactions.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

from pprint import pprint
from scikits.odes import ode
from scikits.odes.sundials.cvode import StatusEnum
from scipy.constants import Avogadro
import math
import numpy as np
import time
import warnings

from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_core
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.species_populations import TempPopulationsLSP
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_utils.util.list import det_dedupe

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class WcSimOdeWarning(UserWarning):
    """ `wc_sim` Ode warning """
    pass


class OdeSubmodel(DynamicSubmodel):
    """ Use a system of ordinary differential equations to predict the dynamics of chemical species in a container

    Attributes:
        ode_time_step (:obj:`float`): the time between ODE solutions
        solver (:obj:`scikits.odes.ode.ode`): the Odes ode solver
        ode_species_ids (:obj:`list`): ids of the species used by this ODE solver
        ode_species_ids_set (:obj:`set`): ids of the species used by this ODE solver
        num_species (:obj:`int`): number of species used by this ODE solver
        populations (:obj:`numpy.ndarray`): pre-allocated numpy array for storing species populations
        non_negative_populations (:obj:`numpy.ndarray`): pre-allocated numpy array for non-negative
            species populations
        zero_populations (:obj:`numpy.ndarray`): pre-allocated numpy array
        rate_of_change_expressions (:obj:`list` of :obj:`list` of :obj:`tuple`): for each species,
            a list of its (coefficient, rate law) pairs
        adjustments (:obj:`dict`): pre-allocated adjustments for passing changes to LocalSpeciesPopulation
        num_species (:obj:`int`): number of species in `ode_species_ids`
        num_right_hand_side_calls (:obj:`int`): number of calls to `right_hand_side` in a call to
            the solver, `self.solver.solve()`
        history_num_right_hand_side_calls (:obj:`list` of :obj:`tuple`): history of number of calls
            to `right_hand_side`
        _last_internal_step (:obj:`float`): the time of the last internal ODE step; for debugging
        testing (:obj:`bool`, optional): if set, produce test output
    """
    ABS_ODE_SOLVER_TOLERANCE = config_multialgorithm['abs_ode_solver_tolerance']
    REL_ODE_SOLVER_TOLERANCE = config_multialgorithm['rel_ode_solver_tolerance']

    # register the message types sent by OdeSubmodel
    messages_sent = [message_types.RunOde]

    # register 'handle_RunOde_msg' to handle RunOde events
    event_handlers = [(message_types.RunOde, 'handle_RunOde_msg')]

    using_solver = False

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, ode_time_step, testing=False):
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
            ode_time_step (:obj:`float`): initial time interval between ODE analyses
            testing (:obj:`bool`, optional): if set, produce test output
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        if ode_time_step <= 0:
            raise MultialgorithmError(f"OdeSubmodel {self.id}: ode_time_step must be positive, but is {ode_time_step}")
        self.ode_time_step = ode_time_step
        self.set_up_ode_submodel()
        self.set_up_optimizations()
        options = {'atol': self.ABS_ODE_SOLVER_TOLERANCE,
                   'rtol': self.REL_ODE_SOLVER_TOLERANCE}
        self.solver = self.create_ode_solver(**options)
        self.num_right_hand_side_calls = 0
        self.history_num_right_hand_side_calls = []
        self.testing = testing
        self._last_internal_step = 0.

    def set_up_ode_submodel(self):
        """ Set up an ODE submodel, including its ODE solver """

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

        # todo: use linear algebra to compute species derivatives, & calc each RL once
        # replace rate_of_change_expressions with a stoichiometric matrix S
        # then, in compute_population_change_rates() compute a vector of reaction rates R and get
        # rates of change of species from R * S
        self.rate_of_change_expressions = []
        for species_id in self.ode_species_ids:
            self.rate_of_change_expressions.append(tmp_coeffs_and_rate_laws[species_id])
        '''
        print('for each species, a list of its (coefficient, rate law) pairs')
        for n, s_list in enumerate(self.rate_of_change_expressions):
            p = []
            for coeff, rl in s_list:
                p.append(f'{coeff} * {rl.id}')
            print(f'species: {n}', ' + '.join(p))
        '''

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        # make fixed set of species ids used by this OdeSubmodel
        self.ode_species_ids_set = set(self.ode_species_ids)
        # pre-allocate dict of adjustments used to pass changes to LocalSpeciesPopulation
        self.adjustments = {species_id: None for species_id in self.ode_species_ids}
        # pre-allocate numpy arrays for populations
        self.num_species = len(self.ode_species_ids)
        self.populations = np.zeros(self.num_species)
        self.non_negative_populations = np.zeros(self.num_species)
        self.zero_populations = np.zeros(self.num_species)

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

    def create_ode_solver(self, **options):
        """ Create a `scikits.odes` ODE solver that uses CVODE

        Args:
            options (:obj:`dict`): options for the solver;
                see https://github.com/bmcage/odes/blob/master/scikits/odes/sundials/cvode.pyx

        Returns:
            :obj:`scikits.odes.ode`: an ODE solver instance

        Raises:
            :obj:`MultialgorithmError`: if the ODE solver cannot be created
        """
        # use CVODE from LLNL's SUNDIALS project (https://computing.llnl.gov/projects/sundials)
        CVODE_SOLVER = 'cvode'
        solver = ode(CVODE_SOLVER, self.right_hand_side, old_api=False, **options)
        if not isinstance(solver, ode):    # pragma: no cover
            raise MultialgorithmError(f"OdeSubmodel {self.id}: scikits.odes.ode() failed")
        # todo: compare and report detailed results: define ODE log file
        run_ode_solver_header = 'mode time time_change rhs_calls elapsed_rt population_0 population_1'.split()
        # todo: compare and report detailed results: fix header & write it to ODE log file
        return solver

    def right_hand_side(self, time, new_species_populations, population_change_rates):
        """ Evaluate population change rates for species modeled by ODE; called by ODE solver

        Args:
            time (:obj:`float`): simulation time
            new_species_populations (:obj:`numpy.ndarray`): populations of all species at time `time`,
                listed in the same order as `self.ode_species_ids`
            population_change_rates (:obj:`numpy.ndarray`): the rate of change of
                `new_species_populations` at time `time`; written by this method

        Returns:
            :obj:`int`: return 0 to indicate success, or 1 to indicate failure;
                but the important side effects are the values in `population_change_rates`
        """
        # `right_hand_side` is called by the CVODE ODE solver

        try:
            self.compute_population_change_rates(time, new_species_populations, population_change_rates)

            self.num_right_hand_side_calls += 1
            # todo: compare and report detailed results: remove 'and self.time == 0'; write summary to ODE debug log file
            if self.testing and self.time == 0:
                # testing
                time_change = time - self._last_internal_step
                summary = ['internal step',
                           f'{time:.4e}',
                           f'{time_change:.4e}',
                           'N/A',
                           'N/A']
                summary.extend([f'{pop:.5e}' for pop in new_species_populations])
                summary.extend([f'{pop_change_rate:.5e}' for pop_change_rate in population_change_rates])
                print('\t'.join(summary))
                self._last_internal_step = time
            return 0

        except Exception as e:
            print(f"OdeSubmodel {self.id}: solver.right_hand_side() failed: '{e}'")
            raise MultialgorithmError(f"OdeSubmodel {self.id}: solver.right_hand_side() failed: '{e}'")
            return 1

    def compute_population_change_rates(self, time, new_species_populations, population_change_rates):
        """ Compute the rate of change of the populations of species used by this ODE

        Args:
            time (:obj:`float`): simulation time
            new_species_populations (:obj:`numpy.ndarray`): populations of all species at time `time`,
                listed in the same order as `self.ode_species_ids`
            population_change_rates (:obj:`numpy.ndarray`): the rate of change of
                `new_species_populations` at time `time`; written by this method
        """

        # Use TempPopulationsLSP to temporarily set the populations of species used by this ODE
        # to the values provided by the ODE solver

        # Replace negative population values with 0 in rate calculation, as recommended by the
        # section "Advice on controlling unphysical negative values" in
        # Hindmarsh, Serban and Reynolds, User Documentation for cvode v5.0.0, 2019
        self.non_negative_populations = np.maximum(self.zero_populations, new_species_populations)
        temporary_populations = dict(zip(self.ode_species_ids, self.non_negative_populations))
        with TempPopulationsLSP(self.local_species_population, temporary_populations):
            for idx in range(self.num_species):
                species_rate_of_change = 0.0
                for coeff, dynamic_rate_law in self.rate_of_change_expressions[idx]:
                    rate = dynamic_rate_law.eval(time)
                    species_rate_of_change += coeff * rate
                population_change_rates[idx] = species_rate_of_change

    def current_species_populations(self):
        """ Obtain the current populations of species modeled by this ODE

        The current populations are written into `self.populations`.
        """
        pops_dict = self.local_species_population.read(self.time, self.ode_species_ids_set, round=False)
        for idx, species_id in enumerate(self.ode_species_ids):
            self.populations[idx] = pops_dict[species_id]

    def run_ode_solver(self):
        """ Run the ODE solver for one WC simulator time step and save the species populations changes """

        if self.testing:
            # testing
            self._last_internal_step = self.time
            start = time.perf_counter()

        ### run the ODE solver ###
        end_time = self.time + self.ode_time_step
        # advance one ode_time_step
        solution_times = [self.time, end_time]
        self.num_right_hand_side_calls = 0
        self.current_species_populations()
        solution = self.solver.solve(solution_times, self.populations)
        if solution.flag != StatusEnum.SUCCESS:   # pragma: no cover
            raise MultialgorithmError(f"OdeSubmodel {self.id}: solver step() error: '{solution.message}' "
                                      f"for time step [{self.time}, {end_time})")

        ### store results in local_species_population ###
        solution_time = solution.values.t[1]
        new_population = solution.values.y[1]
        population_changes = new_population - self.populations
        time_advance = solution_time - self.time
        population_change_rates = population_changes / time_advance

        for idx, species_id in enumerate(self.ode_species_ids):
            self.adjustments[species_id] = population_change_rates[idx]

        if self.testing and self.time == 0:
            print('time_advance:',time_advance)
            if solution.errors.t:
                print('solution.errors.t:', solution.errors.t)
            if solution.errors.y:
                print('solution.errors.y:', solution.errors.y)
            print('population_changes:',population_changes)
            print('population_change_rates:',population_change_rates)
            print('self.ode_species_ids', self.ode_species_ids)
            print('self.adjustments')
            pprint(self.adjustments)
        self.local_species_population.adjust_continuously(self.time, self.adjustments)
        self.history_num_right_hand_side_calls.append(self.num_right_hand_side_calls)
        # todo: compare and report detailed results: write summary to ODE log
        if self.testing:
            end = time.perf_counter()
            elapsed_rt = end - start
            summary = ['external step',
                       f'{self.time:.4e}',
                       f'{time_advance:.4e}',
                       f'{self.num_right_hand_side_calls}',
                       f'{elapsed_rt:.2e}']
            summary.extend([f'{pop:.2e}' for pop in new_population])
            summary.extend([f'{pop_change_rate:.2e}' for pop_change_rate in population_change_rates])
            print('\t'.join(summary))

    def get_info(self):
        """ Get info from CVODE

        Returns:
            :obj:`dict`: the information returned by the solver's `get_info()` call
        """
        return self.solver.get_info()

    ### schedule and handle DES events ###
    def send_initial_events(self):
        """ Send this ODE submodel's initial event """
        self.send_event(0, self, message_types.RunOde())

    def schedule_next_ode_analysis(self):
        """ Schedule the next analysis by this ODE submodel """
        self.send_event(self.ode_time_step, self, message_types.RunOde())

    def handle_RunOde_msg(self, event):
        """ Handle an event containing a RunOde message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_ode_solver()
        self.schedule_next_ode_analysis()
