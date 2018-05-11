""" Take periodic checkpoints in a multialgorithmic simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-08
:Copyright: 2018, Karr Lab
:License: MIT
"""

from wc_sim.core.simulation_checkpoint_object import CheckpointSimulationObject


class MultialgorithmicCheckpointingSimObj(CheckpointSimulationObject):

    def __init__(self, name, checkpoint_period, checkpoint_dir, metadata, local_species_population):
        super().__init__(name, checkpoint_period, checkpoint_dir, metadata, local_species_population)
