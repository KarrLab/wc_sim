""" Take periodic checkpoints in a multialgorithmic simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-08
:Copyright: 2018, Karr Lab
:License: MIT
"""
import numpy
import pandas

from wc_sim.log.checkpoint import Checkpoint
from wc_sim.core.simulation_checkpoint_object import CheckpointSimulationObject, AccessStateObjectInterface
from wc_sim.core.sim_metadata import SimulationMetadata
from wc_sim.multialgorithm.submodels.ssa import SSASubmodel


class MultialgorithmCheckpoint(Checkpoint):
    """ Checkpoint class that holds multialgorithmic checkpoints
    """

    @staticmethod
    def convert_checkpoints(dirname):
        """ Convert the species population in saved checkpoints into a pandas dataframe

        Args:
            dirname (:obj:`str`): directory containing the checkpoint data

        Returns:
            :obj:`pandas.DataFrame`: the species popuation in a simulation checkpoint history
        """
        # create an empty DataFrame
        checkpoints = Checkpoint.list_checkpoints(dirname)
        checkpoint = Checkpoint.get_checkpoint(dirname, time=0)
        species_pop, _ = checkpoint.state
        species_ids = species_pop.keys()
        pred_species_pops = pandas.DataFrame(index=checkpoints, columns=species_ids, dtype=numpy.float64)

        # load the DataFrame
        for time in Checkpoint.list_checkpoints(dirname):
            species_populations, _ = Checkpoint.get_checkpoint(dirname, time=time).state
            for species_id,population in species_populations.items():
                pred_species_pops.loc[time, species_id] = population
        return pred_species_pops


class AccessState(AccessStateObjectInterface):
    """ Obtain checkpoints of a multialgorithm simulation's biological state and random state

    Attributes:
        local_species_population (:obj:`LocalSpeciesPopulation`): provide a simulation's species populations
        dynamic_model (:obj:`DynamicModel`): provide the cell's aggregate state in a simulation
        multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the `MultialgorithmSimulation`
    """

    def __init__(self, local_species_population, dynamic_model, multialgorithm_simulation):
        self.local_species_population = local_species_population
        self.dynamic_model = dynamic_model
        self.multialgorithm_simulation = multialgorithm_simulation

    def get_checkpoint_state(self, time):
        """ Obtain a checkpoint of the biological state

        Returns:
            :obj:`tuple` of (`dict`, `dict`): dictionaries with the species populations and the
                cell's aggregate state, respectively
        """
        return (self.local_species_population.read(time), self.dynamic_model.get_aggregate_state())

    def get_random_state(self):
        """ Obtain a checkpoint of the random state

        Provides a dictionary that maps components of the simulation to their random states, which
        are all instances of `numpy.random.RandomState`

        Returns:
            :obj:`dict`: a dictionary of the random states in the simulation
        """
        random_states = {}
        random_states['local_species_population'] = self.local_species_population.random_state.get_state()
        random_states['submodels'] = {}
        for submodel in self.multialgorithm_simulation.simulation_submodels:
            if isinstance(submodel, SSASubmodel):
                # only SSA submodels use random numbers
                random_states['submodels'][submodel.id] = submodel.random_state.get_state()
        return random_states


class MultialgorithmicCheckpointingSimObj(CheckpointSimulationObject):
    """ A checkpointing simulation object for a multialgorithmic simulatino

    Attributes:
        access_state_object (:obj:`AccessState`): an object that provides checkpoints
    """
    def __init__(self, name, checkpoint_period, checkpoint_dir, metadata, local_species_population,
        dynamic_model, multialgorithm_simulation):
        """ Create a MultialgorithmicCheckpointingSimObj

        Args:
            name (:obj:`str`): name
            checkpoint_period (:obj:`float`): checkpoint period
            checkpoint_dir (:obj:`str`): checkpoint directory
            metadata (:obj:`SimulationMetadata`): metadata
            local_species_population (:obj:`LocalSpeciesPopulation`): the `LocalSpeciesPopulation`
            dynamic_model (:obj:`DynamicModel`): the `DynamicModel`
            multialgorithm_simulation (:obj:`MultialgorithmSimulation`): the `MultialgorithmSimulation`
        """
        self.access_state_object = AccessState(local_species_population, dynamic_model,
            multialgorithm_simulation)
        super().__init__(name, checkpoint_period, checkpoint_dir, metadata, self.access_state_object)
