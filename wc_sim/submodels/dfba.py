import conv_opt

from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.species_populations import TempPopulationsLSP
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_utils.util.list import det_dedupe

class DfbaSubmodel(DynamicSubmodel):

    messages_sent = [message_types.RunFba]

    event_handlers = [(message_types.RunFba, 'handle_RunFba_msg')]

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, dfba_time_step, options=None):

        if not isinstance(dfba_time_step, (float, int)):
            raise MultialgorithmError(f"DfbaSubmodel {self.id}: dfba_time_step must be a number but is "
                                      f"{dfba_time_step}")
        if dfba_time_step <= 0:
            raise MultialgorithmError("dfba_time_step must be positive, but is {}".format(dfba_time_step))
        self.dfba_time_step = dfba_time_step
        self.options = options

        # log initialization data
        self.log_with_time("init: id: {}".format(id))
        self.log_with_time("init: time_step: {}".format(str(dfba_time_step)))

        ### dFBA specific code ###
        # AG: let's use lower_with_under naming for instance variables
        # AG: see https://google.github.io/styleguide/pyguide.html
        self.metabolismProductionReaction = None
        self.exchangedSpecies = None

        self.thermodynamicBounds = None
        self.exchangeRateBounds = None

        self.defaultFbaBound = 1e15
        ### AG: should be size of # of reactions, I believe
        self.reactionFluxes = np.zeros(0)

        self.set_up_dfba_submodel()
        self.set_up_optimizations()

    def set_up_dfba_submodel(self):
        """ Set up a DFBA submodel, by converting to a linear programming matrix """

        fba_species = []
        for idx, rxn in enumerate(self.reactions):
            for species_coefficient in rxn.participants:
                species_id = species_coefficient.species.gen_id()
                fba_species.append(species_id)
        self.fba_species_ids = det_dedupe(fba_species)

        ### dFBA specific code ###
        # AG: TODO
        # Reuse my code to convert reactions to conv_opt variables
        # Create a dictionary that maps variables to reactions
        # Create a dictionary that maps species to a dictionary that maps variables to species stoichiometric coefficient
        # Reuse my code to set up the objective function in conv_opt format

        self.schedule_next_FBA_analysis()

    def schedule_next_fba_analysis(self):
        """ Schedule the next analysis by this FBA submodel.
        """
        self.send_event(self.dfba_time_step, self, message_types.RunFba())

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        # make fixed set of species ids used by this FbaSubmodel
        self.fba_species_ids_set = set(self.fba_species_ids)
        # pre-allocate dict of adjustments used to pass changes to LocalSpeciesPopulation
        self.adjustments = {species_id: None for species_id in self.fba_species_ids}
        # pre-allocate numpy arrays for populations
        self.num_species = len(self.fba_species_ids)
        self.populations = np.zeros(self.num_species)

    def current_species_populations(self):
        """ Obtain the current populations of species modeled by this FBA
        The current populations are written into `self.populations`.
        """
        pops_dict = self.local_species_population.read(self.time, self.fba_species_ids_set, round=False)
        for idx, species_id in enumerate(self.fba_species_ids):
            self.populations[idx] = pops_dict[species_id]

    def run_fba_solver(self):
        """ Run the FBA solver for one time step """

        ### dFBA specific code ###
        # AG: TODO
        # AG: do the bounds come from rate laws? I thought they would come from Reaction flux bounds in
        # wc_lang.core.Reaction.flux_min and wc_lang.core.Reaction.flux_max
        # Eval rate laws and set as bounds
        # Set bounds of exchange reactions
        # Maximize objective function and retrieve the reaction fluxes
        # If no solution is found, raise error

        self.current_species_populations()
        # Calculate the adjustment for each species as sum over reactions of reaction flux * stoichiometry / dfba_time_step
        # Check that the adjustment will not cause a species population to become negative
        # AG: I think that negative populations will need to generate an error
        # the only other possibility would be to reduce the time step such that populations
        # stay positive, subject to a minimum time step
        # If it becomes negative, then change the adjustment so that the species population is zero

        ### store results in local_species_population ###
        self.local_species_population.adjust_continuously(self.time, self.adjustments)

        # flush expressions that depend on species and reactions modeled by this FBA submodel from cache
        # AG: we can rename "ode_flush_after_populations_change" to
        # "continuous_submodel_flush_after_populations_change" and use it both here and in odes.py
        self.dynamic_model.fba_flush_after_populations_change(self.id) # will add this method to dynamic_components.py

    def handle_RunFba_msg(self):
        """ Handle an event containing a RunFba message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_fba_solver()
        self.schedule_next_fba_analysis()    
