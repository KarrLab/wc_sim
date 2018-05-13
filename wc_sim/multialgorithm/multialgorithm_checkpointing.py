""" Take periodic checkpoints in a multialgorithmic simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-08
:Copyright: 2018, Karr Lab
:License: MIT
"""

from wc_sim.core.simulation_checkpoint_object import CheckpointSimulationObject, AccessStateObjectInterface


class AccessStateObject(AccessStateObjectInterface):
    """ Get a checkpoint for a multialgorithm simulation

    Attributes:
        local_species_population (:obj:`LocalSpeciesPopulation`): provide a simulation's species populations
        dynamic_model (:obj:`DynamicModel`): provide the cell's aggregate state in a simulation
    """

    def __init__(self, local_species_population, dynamic_model):
        self.local_species_population = local_species_population
        self.dynamic_model = dynamic_model

    def get_checkpoint_state(self, time):
        """ Obtain a checkpoint

        Returns:
            :obj:`tuple` of (`dict`, `dict`): dictionaries with the species populations and the
                cell's aggregate state
        """
        return (self.local_species_population.read(time), self.dynamic_model.get_aggregate_state())


class MultialgorithmicCheckpointingSimObj(CheckpointSimulationObject):
    """ A checkpointing simulation object for a multialgorithmic simulatino

    Attributes:
        access_state_object (:obj:`AccessStateObject`): an object that provides checkpoints
    """

    def __init__(self, name, checkpoint_period, checkpoint_dir, metadata, local_species_population,
        dynamic_model):
        self.access_state_object = AccessStateObject(local_species_population, dynamic_model)
        super().__init__(name, checkpoint_period, checkpoint_dir, metadata, self.access_state_object)
