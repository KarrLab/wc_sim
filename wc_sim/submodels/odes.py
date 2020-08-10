""" A submodel that uses a system of ordinary differential equations (ODEs) to model a set of reactions.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-10-12
:Copyright: 2018, Karr Lab
:License: MIT
"""

from scikits.odes import ode
from scikits.odes.sundials.cvode import StatusEnum
from scipy.constants import Avogadro
import math
import numpy as np
import time
import warnings

from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.species_populations import TempPopulationsLSP
from wc_sim.submodels.dynamic_submodel import ContinuousTimeSubmodel

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class WcSimOdeWarning(UserWarning):
    """ `wc_sim` Ode warning """
    pass


class OdeSubmodel(ContinuousTimeSubmodel):
    """ Use a system of ordinary differential equations to predict the dynamics of chemical species in a container

    Attributes:
        solver (:obj:`scikits.odes.ode.ode`): the Odes ode solver
        non_negative_populations (:obj:`numpy.ndarray`): pre-allocated numpy array for non-negative
            species populations
        zero_populations (:obj:`numpy.ndarray`): pre-allocated numpy array
        rate_of_change_expressions (:obj:`list` of :obj:`list` of :obj:`tuple`): for each species,
            a list of its (coefficient, rate law) pairs
    """
    ABS_ODE_SOLVER_TOLERANCE = config_multialgorithm['abs_ode_solver_tolerance']
    REL_ODE_SOLVER_TOLERANCE = config_multialgorithm['rel_ode_solver_tolerance']

    # register the message types sent by OdeSubmodel
    messages_sent = [message_types.RunOde]

    # register 'handle_RunOde_msg' to handle RunOde events
    event_handlers = [(message_types.RunOde, 'handle_RunOde_msg')]

    time_step_message = message_types.RunOde

    using_solver = False

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, ode_time_step, options=None):
        """ Initialize an ODE submodel instance

        Args:
            id (:obj:`str`): unique id of this dynamic ODE submodel
            dynamic_model (:obj: `DynamicModel`): the simulation's central coordinator
            reactions (:obj:`list` of `wc_lang.Reaction`): the reactions modeled by this ODE submodel
            species (:obj:`list` of `wc_lang.Species`): the species that participate in the reactions
                modeled by this ODE submodel
            dynamic_compartments (:obj: `dict`): `DynamicCompartment`s, keyed by id, that contain
                species which participate in reactions that this ODE submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                ODE submodel's species population
            ode_time_step (:obj:`float`): time interval between ODE analyses
            options (:obj:`dict`, optional): ODE submodel options
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population, ode_time_step, options)
        self.set_up_ode_submodel()
        self.set_up_optimizations()
        ode_solver_options = {'atol': self.ABS_ODE_SOLVER_TOLERANCE,
                              'rtol': self.REL_ODE_SOLVER_TOLERANCE}
        if options is not None and 'tolerances' in options:
            if 'atol' in options['tolerances']:
                ode_solver_options['atol'] = options['tolerances']['atol']
            if 'rtol' in options['tolerances']:
                ode_solver_options['rtol'] = options['tolerances']['rtol']
        self.solver = self.create_ode_solver(**ode_solver_options)

    def set_up_ode_submodel(self):
        """ Set up an ODE submodel, including its ODE solver """
        # disable locking temporarily
        # self.get_solver_lock()
        self.set_up_continuous_time_submodel()

        # this is optimal, but costs O(|self.reactions| * |rxn.participants|)
        tmp_coeffs_and_rate_laws = {species_id:[] for species_id in self.species_ids}
        for idx, rxn in enumerate(self.reactions):
            rate_law_id = rxn.rate_laws[0].id
            dynamic_rate_law = self.dynamic_model.dynamic_rate_laws[rate_law_id]
            for species_coefficient in rxn.participants:
                species_id = species_coefficient.species.gen_id()
                tmp_coeffs_and_rate_laws[species_id].append((species_coefficient.coefficient,
                                                             dynamic_rate_law))

        # TODO(Arthur): use vector algebra to compute species derivatives, & calc each RL once
        # replace rate_of_change_expressions with a stoichiometric matrix S
        # then, in compute_population_change_rates() compute a vector of reaction rates R and get
        # rates of change of species from R * S
        self.rate_of_change_expressions = []
        for species_id in self.species_ids:
            self.rate_of_change_expressions.append(tmp_coeffs_and_rate_laws[species_id])

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        self.set_up_continuous_time_optimizations()
        self.non_negative_populations = np.zeros(self.num_species)
        self.zero_populations = np.zeros(self.num_species)

    def get_solver_lock(self):
        """ Acquire the solver lock

        Raises:
            :obj:`DynamicMultialgorithmError`: if the solver lock cannot be acquired
        """
        cls = self.__class__
        if not cls.using_solver:
            cls.using_solver = True
            return True
        else:
            raise DynamicMultialgorithmError(self.time, "OdeSubmodel {}: cannot get_solver_lock".format(self.id))

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
        return solver

    def right_hand_side(self, time, new_species_populations, population_change_rates):
        """ Evaluate population change rates for species modeled by ODE; called by ODE solver

        This is called by the CVODE ODE solver.

        Args:
            time (:obj:`float`): simulation time
            new_species_populations (:obj:`numpy.ndarray`): estimated populations of all species at
                time `time`, provided by the ODE solver;
                listed in the same order as `self.species_ids`
            population_change_rates (:obj:`numpy.ndarray`): the rate of change of
                `new_species_populations` at time `time`; written by this method

        Returns:
            :obj:`int`: return 0 to indicate success, or 1 to indicate failure;
                but the important side effects are the values in `population_change_rates`

        Raises:
            :obj:`DynamicMultialgorithmError`: if this method raises any exception
        """
        try:
            self.compute_population_change_rates(time, new_species_populations, population_change_rates)
            return 0

        except Exception as e:
            # the CVODE ODE solver requires that RHS return 1 if it fails, but raising an
            # exception will provide more error information
            raise DynamicMultialgorithmError(self.time,
                                             f"OdeSubmodel {self.id}: solver.right_hand_side() failed: '{e}'")
            return 1    # pragma: no cover

    def compute_population_change_rates(self, time, new_species_populations, population_change_rates):
        """ Compute the rate of change of the populations of species used by this ODE

        Args:
            time (:obj:`float`): simulation time
            new_species_populations (:obj:`numpy.ndarray`): populations of all species at time `time`,
                listed in the same order as `self.species_ids`
            population_change_rates (:obj:`numpy.ndarray`): the rate of change of
                `new_species_populations` at time `time`; written by this method
        """
        # Use TempPopulationsLSP to temporarily set the populations of species used by this ODE
        # to the values provided by the ODE solver

        # Replace negative population values with 0 in rate calculation, as recommended by the
        # section "Advice on controlling unphysical negative values" in
        # Hindmarsh, Serban and Reynolds, User Documentation for cvode v5.0.0, 2019
        self.non_negative_populations = np.maximum(self.zero_populations, new_species_populations)
        temporary_populations = dict(zip(self.species_ids, self.non_negative_populations))
        with TempPopulationsLSP(self.local_species_population, temporary_populations):

            # flush expressions that depend on species and reactions modeled by this ODE submodel from cache
            self.dynamic_model.continuous_submodel_flush_after_populations_change(self.id)

            for idx in range(self.num_species):
                species_rate_of_change = 0.0
                for coeff, dynamic_rate_law in self.rate_of_change_expressions[idx]:
                    rate = dynamic_rate_law.eval(time)
                    species_rate_of_change += coeff * rate
                population_change_rates[idx] = species_rate_of_change

    def run_ode_solver(self):
        """ Run the ODE solver for one WC simulator time step and save the species populations changes

        Raises:
            :obj:`DynamicMultialgorithmError`: if the CVODE ODE solver indicates an error
        """
        ### run the ODE solver ###
        end_time = self.time + self.time_step
        # advance one ode_time_step
        solution_times = [self.time, end_time]
        self.current_species_populations()
        solution = self.solver.solve(solution_times, self.populations)
        if solution.flag != StatusEnum.SUCCESS:   # pragma: no cover
            raise DynamicMultialgorithmError(self.time, f"OdeSubmodel {self.id}: solver step() error: "
                                                        f"'{solution.message}' "
                                                        f"for time step [{self.time}, {end_time})")

        ### store results in local_species_population ###
        solution_time = solution.values.t[1]
        new_population = solution.values.y[1]
        population_changes = new_population - self.populations
        time_advance = solution_time - self.time
        population_change_rates = population_changes / time_advance
        for idx, species_id in enumerate(self.species_ids):
            self.adjustments[species_id] = population_change_rates[idx]
        self.local_species_population.adjust_continuously(self.time, self.adjustments)

        # flush expressions that depend on species and reactions modeled by this ODE submodel from cache
        self.dynamic_model.continuous_submodel_flush_after_populations_change(self.id)

    def get_info(self):
        """ Get info from CVODE

        Returns:
            :obj:`dict`: the information returned by the solver's `get_info()` call
        """
        return self.solver.get_info()

    ### handle DES events ###
    def handle_RunOde_msg(self, event):
        """ Handle an event containing a RunOde message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_ode_solver()
        self.schedule_next_periodic_analysis()
