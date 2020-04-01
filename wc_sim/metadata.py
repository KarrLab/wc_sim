""" Store metadata about a WC simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-03-29
:Copyright: 2020, Karr Lab
:License: MIT
"""

import os
from dataclasses import dataclass
import warnings

from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.sim_config import WCSimulationConfig
from wc_utils.util.git import get_repo_metadata, RepoMetadataCollectionType, RepositoryMetadata
from wc_utils.util.misc import EnhancedDataClass
from wc_sim.multialgorithm_errors import MultialgorithmWarning


@dataclass
class WCSimulationMetadata(EnhancedDataClass):
    """ Represent a WC simulation's metatdata

    Attributes:
        wc_sim_config (:obj:`WCSimulationConfig`): a Whole-cell simulation configuration
        wc_simulator_repo (:obj:`RepositoryMetadata`): Git repository repo metadata about the WC simulator
        wc_model_repo (:obj:`RepositoryMetadata`): Git repository repo metadata about the WC repo storing
            the WC model being simulated
    """

    wc_sim_config: WCSimulationConfig
    wc_simulator_repo: RepositoryMetadata = None
    wc_model_repo: RepositoryMetadata = None

    def __setattr__(self, name, value):
        """ Validate an attribute when it is changed """
        try:
            super().__setattr__(name, value)
        except TypeError as e:
            raise MultialgorithmError(e)

    def __post_init__(self):
        self.wc_simulator_repo, _ = get_repo_metadata(path=__file__,
                                                      repo_type=RepoMetadataCollectionType.SCHEMA_REPO)

    def set_wc_model_repo(self, model_path):
        """ Set the value of `wc_model_repo` if it can be obtained from `model_path`

        Args:
            model_path (:obj:`str`): path to a file in the model's Git repository

        Warns:
            :obj:`MultialgorithmWarning`: if obj:`path` is not a path in a Git repository,
                or if the repository is not suitable for gathering metadata
        """
        try:
            self.wc_model_repo, _ = get_repo_metadata(path=model_path,
                                                      repo_type=RepoMetadataCollectionType.SCHEMA_REPO)
        except ValueError as e:
            warnings.warn(f"Cannot obtain metadata for git repo containing model at '{model_path}': {e}",
                          MultialgorithmWarning)

    @staticmethod
    def get_pathname(dirname):
        """ See docstring in :obj:`EnhancedDataClass`
        """
        return os.path.join(dirname, 'wc_simulation_metadata.pickle')
